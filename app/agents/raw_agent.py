"""
OpenAI SDK implementation of the assessment recommender agent.

Two LLM calls per /chat turn (maximum):
  1. Turn classifier  — gpt-4o-mini, strict JSON schema output
  2. Composer / compare — gpt-4o-mini, free-form text

The classifier re-derives ALL state from the full conversation history on every
call. There is no server-side session store.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import numpy as np

from app.data_loader import CatalogStore, fuzzy_lookup, semantic_search, semantic_search_with_scores
from app.models import (
    CatalogItem,
    ChatResponse,
    Message,
    Recommendation,
    TurnClassification,
)
from app.prompts import (
    CLARIFY_SYSTEM,
    CLARIFY_USER_TEMPLATE,
    CLASSIFIER_SYSTEM,
    CLASSIFIER_USER_TEMPLATE,
    COMPARE_SYSTEM,
    COMPARE_USER_TEMPLATE,
    COMPOSER_SYSTEM,
    COMPOSER_USER_TEMPLATE,
    REFUSAL_TEMPLATE,
)

logger = logging.getLogger(__name__)

# JSON schema passed to gpt-4o-mini as response_format — enforces structure
# JSON schema passed to gpt-4o-mini as response_format.
# IMPORTANT: `reasoning` is intentionally the FIRST property.  OpenAI generates
# JSON object fields in schema order, so placing it first forces the model to
# write its chain-of-thought ledger before it commits to any extracted value.
_CLASSIFIER_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "turn_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                # ── CoT scratchpad — written first, never forwarded to users ──
                "reasoning": {"type": "string"},
                # ── Classification output ─────────────────────────────────────
                "turn_type": {
                    "type": "string",
                    "enum": [
                        "new_info", "refine_add", "refine_remove",
                        "refine_disambiguate", "compare_request",
                        "closing_confirm", "off_topic_refusal",
                    ],
                },
                "role_context":       {"type": "string"},
                "seniority":          {"type": "string"},
                "skills":             {"type": "array", "items": {"type": "string"}},
                "locale":             {"type": "string"},
                "purpose":            {"type": "string"},
                "named_removals":     {"type": "array", "items": {"type": "string"}},
                "compare_targets":    {"type": "array", "items": {"type": "string"}},
                "explicit_adds":      {"type": "array", "items": {"type": "string"}},
                "current_shortlist":  {"type": "array", "items": {"type": "string"}},
                "has_enough_context": {"type": "boolean"},
            },
            "required": [
                # reasoning must come first in the required list to match
                # property order — some OpenAI runtime versions respect this.
                "reasoning",
                "turn_type", "role_context", "seniority", "skills", "locale",
                "purpose", "named_removals", "compare_targets", "explicit_adds",
                "current_shortlist", "has_enough_context",
            ],
            "additionalProperties": False,
        },
    },
}

_OPQ32R_NAME = "Occupational Personality Questionnaire OPQ32r"

# Keyword-boost constants — calibrated so a single exact name-match can lift
# an item ~0.20 above its cosine score, but can never dominate a genuinely
# high-similarity result (cosine scores for strong matches sit around 0.65-0.80).
_KW_BONUS_NAME = 0.20   # skill string appears verbatim in the item name
_KW_BONUS_DESC = 0.08   # skill string appears verbatim in the item description
_KW_INJECT_BASE = 0.38  # base score for keyword-only injections (above gap threshold)

# Calibrated from calibrate_retrieval.py against 12 representative queries.
# Scores below this indicate the top FAISS result is a weak stretch — the
# composer will name the gap explicitly. Set at p25(top-1 scores) - 0.05 = 0.40.
# Do NOT use this to filter out-of-scope queries — the classifier handles that.
_LOW_SIMILARITY_THRESHOLD = 0.40


class RawAgentService:
    def __init__(self, store: CatalogStore, openai_client):
        self._store = store
        self._client = openai_client

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    # Maximum clarifying questions before recommending anyway (8-turn cap)
    _CLARIFY_BUDGET = 2

    def process_turn(self, messages: list[Message]) -> ChatResponse:
        classification = self._classify(messages)
        turn = classification.turn_type

        if turn == "off_topic_refusal":
            return self._handle_refusal(classification)
        if turn == "closing_confirm":
            return self._handle_confirm(classification)
        if turn == "compare_request":
            return self._handle_compare(classification, messages)

        if not classification.has_enough_context:
            spent = self._clarification_turns_spent(messages, classification)
            if spent >= self._CLARIFY_BUDGET:
                # Soft budget hit — recommend with a caveat rather than asking again
                missing = self._missing_dimension(classification)
                caveat = (
                    f"proceeding without confirmed {missing} — acknowledge the "
                    f"uncertainty briefly and invite the user to refine afterwards"
                )
                return self._handle_recommend(classification, caveat=caveat)
            return self._handle_clarify(classification)

        if turn in {"new_info", "refine_add", "refine_remove", "refine_disambiguate"}:
            return self._handle_refine(classification)

        return self._handle_recommend(classification)

    # ------------------------------------------------------------------
    # Step 1: classify the latest turn
    # ------------------------------------------------------------------

    def _classify(self, messages: list[Message]) -> TurnClassification:
        history_text = "\n".join(
            f"[{m.role.upper()}]: {m.content}" for m in messages
        )
        user_prompt = CLASSIFIER_USER_TEMPLATE.format(history=history_text)

        response = self._client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format=_CLASSIFIER_JSON_SCHEMA,
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
        )
        data = json.loads(response.choices[0].message.content)
        return TurnClassification(**data)

    # ------------------------------------------------------------------
    # Branch handlers
    # ------------------------------------------------------------------

    def _handle_refusal(self, c: TurnClassification) -> ChatResponse:
        # Change this line:
        reply = REFUSAL_TEMPLATE
        
        shortlist = self._rebuild_shortlist(c.current_shortlist)
        return ChatResponse(
            reply=self._attach_shortlist_footer(reply, shortlist),
            recommendations=shortlist,
            end_of_conversation=False,
        )

    def _handle_confirm(self, c: TurnClassification) -> ChatResponse:
        shortlist = self._rebuild_shortlist(c.current_shortlist)
        reply = (
            "Confirmed — that's your final battery. Good luck with the hiring process."
            if shortlist
            else "Got it. Let me know if you need anything else."
        )
        return ChatResponse(
            reply=self._attach_shortlist_footer(reply, shortlist),
            recommendations=shortlist,
            end_of_conversation=True,
        )

    def _handle_clarify(self, c: TurnClassification) -> ChatResponse:
        missing = self._missing_dimension(c)
        user_prompt = CLARIFY_USER_TEMPLATE.format(
            role_context=c.role_context or "(unknown)",
            seniority=c.seniority or "(unknown)",
            purpose=c.purpose or "(unknown)",
            locale=c.locale or "(not stated)",
            missing=missing,
        )
        response = self._client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            messages=[
                {"role": "system", "content": CLARIFY_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
        )
        question = response.choices[0].message.content.strip()
        return ChatResponse(
            reply=question,
            recommendations=[],
            end_of_conversation=False,
        )

    def _handle_compare(
        self, c: TurnClassification, messages: list[Message]
    ) -> ChatResponse:
        targets = c.compare_targets
        if len(targets) < 2:
            shortlist = self._rebuild_shortlist(c.current_shortlist)
            return ChatResponse(
                reply=self._attach_shortlist_footer(
                    "I can compare two named assessments once you tell me both product names.",
                    shortlist,
                ),
                recommendations=shortlist,
                end_of_conversation=False,
            )

        item_a = fuzzy_lookup(self._store, targets[0])
        item_b = fuzzy_lookup(self._store, targets[1])

        if item_a is None or item_b is None:
            missing = targets[0] if item_a is None else targets[1]
            shortlist = self._rebuild_shortlist(c.current_shortlist)
            return ChatResponse(
                reply=self._attach_shortlist_footer(
                    (
                    f'I couldn\'t find "{missing}" in the SHL catalog. '
                    "Could you check the name and try again?"
                    ),
                    shortlist,
                ),
                recommendations=shortlist,
                end_of_conversation=False,
            )

        question = messages[-1].content
        user_prompt = COMPARE_USER_TEMPLATE.format(
            question=question,
            name_a=item_a.name,
            desc_a=item_a.description,
            name_b=item_b.name,
            desc_b=item_b.description,
        )
        response = self._client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": COMPARE_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
        )
        reply = response.choices[0].message.content.strip()
        shortlist = self._rebuild_shortlist(c.current_shortlist)
        return ChatResponse(
            reply=self._attach_shortlist_footer(reply, shortlist),
            recommendations=shortlist,
            end_of_conversation=False,
        )

    def _handle_recommend(
        self, c: TurnClassification, caveat: str = ""
    ) -> ChatResponse:
        candidates, defaults_added, catalog_gaps, score_map = self._retrieve(c)

        # Surface the soft-budget caveat through the existing catalog_gaps channel
        if caveat:
            catalog_gaps = [caveat] + catalog_gaps

        # Full description is sent — SHL descriptions are ~100-400 chars each
        # so 10 items is well within the gpt-4o-mini context window. Truncating
        # to 100 chars was silently dropping the most differentiating details
        # (which ability domain, which job family, what the test measures).
        shortlist_text = "\n".join(
            f"  {i+1}. {item.name} [{item.test_type}]\n"
            f"      Keys: {', '.join(item.keys_raw) if item.keys_raw else 'N/A'}\n"
            f"      Duration: {item.duration_display or 'N/A'} | "
            f"Languages: {', '.join(item.languages[:3]) if item.languages else 'N/A'}\n"
            f"      Description: {item.description}"
            for i, item in enumerate(candidates)
        )
        user_prompt = COMPOSER_USER_TEMPLATE.format(
            role_context=c.role_context or "unspecified",
            seniority=c.seniority or "unspecified",
            skills=", ".join(c.skills) if c.skills else "none specified",
            purpose=c.purpose or "unspecified",
            locale=c.locale or "not specified",
            turn_type=c.turn_type,
            count=len(candidates),
            shortlist_text=shortlist_text,
            defaults_added=", ".join(defaults_added) if defaults_added else "none",
            catalog_gaps=", ".join(catalog_gaps) if catalog_gaps else "none",
        )
        response = self._client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[
                {"role": "system", "content": COMPOSER_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
        )
        reply = response.choices[0].message.content.strip()

        recommendations = [
            Recommendation(
                name=item.name,
                url=item.url,
                test_type=item.test_type,
                keys=item.keys_raw,
                duration=item.duration_display,
                languages=item.languages,
                score=score_map.get(item.entity_id),
            )
            for item in candidates
        ]
        self._assert_no_hallucinated_urls(recommendations)
        reply = self._attach_shortlist_footer(reply, recommendations)

        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=False,
        )

    def _handle_refine(self, c: TurnClassification) -> ChatResponse:
        return self._handle_recommend(c)

    # ------------------------------------------------------------------
    # Retrieval pipeline
    # ------------------------------------------------------------------

    def _retrieve(
        self, c: TurnClassification
    ) -> tuple[list[CatalogItem], list[str], list[str], dict[str, float]]:
        """Returns (candidates, defaults_added, catalog_gaps, score_map).

        score_map: entity_id → combined cosine+keyword score (for testing/tuning).
        Items added via shortlist rebuild or explicit_adds get score 0.0.

        For refine turns with an established shortlist the existing items are
        preserved as the starting point so the user never loses their list.
        For new_info (or when there is no prior shortlist) a fresh FAISS search
        is performed.
        """
        catalog_gaps: list[str] = []
        score_map: dict[str, float] = {}
        is_refine = c.turn_type in {"refine_add", "refine_remove", "refine_disambiguate"}

        if is_refine and c.current_shortlist:
            # Rebuild the standing shortlist from history
            candidates: list[CatalogItem] = []
            for name in c.current_shortlist:
                item = fuzzy_lookup(self._store, name, threshold=70)
                if item:
                    candidates.append(item)

            # Apply the latest removals against the preserved list
            candidates = self._apply_removals(candidates, c.named_removals)

            # For refine_add: supplement with fresh FAISS results to fill gaps.
            # Fetch k=20 so there are enough candidates after quality filtering.
            if c.turn_type == "refine_add" and len(candidates) < 10:
                query_vec = self._embed(self._build_persona_query(c))
                scored = semantic_search_with_scores(self._store, query_vec, k=20)
                scored = self._inject_keyword_matches(scored, c.skills)
                scored = self._apply_keyword_boost(scored, c.skills)
                for item, score in scored:
                    if item not in candidates and score >= _LOW_SIMILARITY_THRESHOLD:
                        candidates.append(item)
                        score_map[item.entity_id] = round(score, 4)
                    if len(candidates) >= 10:
                        break

        else:
            # Fresh retrieval for new_info or when no prior shortlist exists.
            # k=20 gives a wide enough pool so quality filtering still leaves
            # up to 10 good results rather than padding with weak ones.
            query_vec = self._embed(self._build_persona_query(c))
            scored_candidates = semantic_search_with_scores(self._store, query_vec, k=20)

            # Record the raw top cosine score BEFORE boosting — this is the
            # true signal for gap detection (keyword boost doesn't mean the
            # catalog has a purpose-built test for the requested skill).
            top_raw_score = scored_candidates[0][1] if scored_candidates else None

            # Hybrid step 1: inject keyword matches the embedding search missed.
            scored_candidates = self._inject_keyword_matches(scored_candidates, c.skills)

            # Hybrid step 2: boost and re-sort by combined score.
            scored_candidates = self._apply_keyword_boost(scored_candidates, c.skills)

            # Dynamic quality filter: keep items at or above the threshold.
            # Guarantee at least 3 results even for niche queries so the agent
            # never returns an empty shortlist when the catalog has any match.
            above = [(item, s) for item, s in scored_candidates if s >= _LOW_SIMILARITY_THRESHOLD]
            if len(above) < 3:
                above = scored_candidates[:3]   # top-3 fallback, no padding below that
            scored_candidates = above

            # Build score map before reranking changes order
            for item, s in scored_candidates:
                score_map[item.entity_id] = round(s, 4)

            candidates = [item for item, _score in scored_candidates]
            candidates = self._rerank(candidates, c)
            candidates = self._apply_removals(candidates, c.named_removals)

            if top_raw_score is not None and top_raw_score < _LOW_SIMILARITY_THRESHOLD:
                signal = c.skills[0] if c.skills else (c.role_context or "this request")
                gap_note = f'no strong match for "{signal}" in the catalog'
                if candidates:
                    gap_note += f"; using {candidates[0].name} as the closest substitute"
                catalog_gaps.append(gap_note)

        # Explicit adds and gap detection apply regardless of turn type
        for add_name in c.explicit_adds:
            match = fuzzy_lookup(self._store, add_name)
            if match and match not in candidates:
                candidates.insert(0, match)
            elif match is None:
                catalog_gaps.append(add_name)

        candidates, defaults_added = self._apply_default_bundling(candidates, c)

        if not catalog_gaps:
            catalog_gaps = self._detect_skill_gaps(c.skills, candidates)

        return candidates[:10], defaults_added, catalog_gaps, score_map

    def _build_persona_query(self, c: TurnClassification) -> str:
        parts = []
        if c.role_context:
            parts.append(c.role_context)
        if c.seniority:
            parts.append(c.seniority)
        if c.skills:
            parts.append("skills: " + ", ".join(c.skills))
        if c.purpose:
            parts.append(c.purpose + " assessment")
        if c.locale:
            parts.append(c.locale)
        return " ".join(parts) or "general professional assessment"

    def _embed(self, text: str) -> np.ndarray:
        response = self._client.embeddings.create(
            model="text-embedding-3-small",
            input=[text],
        )
        vec = np.array(response.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _rerank(
        self, candidates: list[CatalogItem], c: TurnClassification
    ) -> list[CatalogItem]:
        """Boost items matching seniority and locale to the front."""
        if not c.seniority and not c.locale:
            return self._rank_by_relevance(candidates, c)

        boosted, rest = [], []
        for item in candidates:
            level_ok = (
                not c.seniority
                or any(c.seniority.lower() in lvl.lower() for lvl in item.job_levels)
            )
            locale_ok = (
                not c.locale
                or not item.languages  # unlisted → assume compatible
                or any(c.locale.lower() in lang.lower() for lang in item.languages)
            )
            (boosted if level_ok and locale_ok else rest).append(item)

        return self._rank_by_relevance(boosted + rest, c)

    def _rank_by_relevance(
        self, candidates: list[CatalogItem], c: TurnClassification
    ) -> list[CatalogItem]:
        role_terms = [
            term for term in re.findall(r"[A-Za-z0-9+#.-]+", c.role_context.lower())
            if len(term) > 2
        ]
        skill_terms = [
            term for term in (skill.lower() for skill in c.skills)
            if len(term) > 2
        ]
        ordered = []
        for item in candidates:
            text = " ".join([
                item.name,
                item.description,
                " ".join(item.job_levels),
                " ".join(item.languages),
            ]).lower()
            score = 0
            if c.seniority and any(c.seniority.lower() in lvl.lower() for lvl in item.job_levels):
                score += 3
            if c.locale and (not item.languages or any(c.locale.lower() in lang.lower() for lang in item.languages)):
                score += 2
            for term in role_terms:
                if term in text:
                    score += 1
            for term in skill_terms:
                if term in text:
                    score += 1
            ordered.append((score, item))
        ordered.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _score, item in ordered]

    def _apply_removals(
        self, candidates: list[CatalogItem], removals: list[str]
    ) -> list[CatalogItem]:
        if not removals:
            return candidates
        return [
            item for item in candidates
            if not self._is_named_removal(item.name, removals)
        ]

    def _is_named_removal(self, item_name: str, removals: list[str]) -> bool:
        name_lower = item_name.lower()
        for removal in removals:
            r = removal.lower()
            if r in name_lower or name_lower in r:
                return True
            match = fuzzy_lookup(self._store, removal, threshold=75)
            if match and match.name.lower() == name_lower:
                return True
        return False

    def _apply_default_bundling(
        self, candidates: list[CatalogItem], c: TurnClassification
    ) -> tuple[list[CatalogItem], list[str]]:
        """
        Add a role-appropriate personality/behaviour measure when none is present.

        Hierarchy (§5.6 of PROJECT_CONTEXT.md):
          safety/industrial → DSI or Manufacturing Safety bundle
          sales             → OPQ MQ Sales Report (+ OPQ32r fallback)
          contact centre    → Entry Level Customer Serv bundle
          general           → OPQ32r
        """
        if any("P" in item.test_type for item in candidates):
            return candidates, []

        role = c.role_context.lower()
        if any(kw in role for kw in ("safety", "industrial", "plant", "operator")):
            targets = [
                "Dependability and Safety Instrument (DSI)",
                "Manufac. & Indust. - Safety & Dependability 8.0",
            ]
        elif "sales" in role:
            targets = ["OPQ MQ Sales Report", _OPQ32R_NAME]
        elif any(kw in role for kw in ("contact cent", "customer serv", "call cent")):
            targets = ["Entry Level Customer Serv-Retail & Contact Center"]
        else:
            targets = [_OPQ32R_NAME]

        for target in targets:
            match = fuzzy_lookup(self._store, target, threshold=70)
            if match and match not in candidates:
                candidates.append(match)
                return candidates, [match.name]
            if match in candidates:
                return candidates, []

        return candidates, []

    def _detect_skill_gaps(
        self, skills: list[str], candidates: list[CatalogItem]
    ) -> list[str]:
        if not skills:
            return []
        candidate_text = " ".join(
            (item.name + " " + item.description).lower() for item in candidates
        )
        return [s for s in skills if s.lower() not in candidate_text]

    def _inject_keyword_matches(
        self,
        scored: list[tuple[CatalogItem, float]],
        skills: list[str],
    ) -> list[tuple[CatalogItem, float]]:
        """
        Scan the full catalog for items whose name or description contains any
        explicit skill keyword and inject those not already present in `scored`.

        Pure cosine search with k=20 can miss niche acronyms and version numbers
        (e.g. ".NET Framework 4.5", "AWS Lambda") whose embeddings sit far from
        the generic role query.  This catches them and inserts them at a neutral
        baseline score so subsequent boosting can elevate them appropriately.

        The catalog is only 377 items so an O(n) linear scan is negligible here.
        """
        if not skills:
            return scored

        skill_terms = [s.lower() for s in skills]
        present_ids = {item.entity_id for item, _ in scored}

        for item in self._store.items:
            if item.entity_id in present_ids:
                continue  # already returned by FAISS — don't duplicate
            name_lower = item.name.lower()
            desc_lower = item.description.lower()
            if any(term in name_lower or term in desc_lower for term in skill_terms):
                scored.append((item, _KW_INJECT_BASE))
                present_ids.add(item.entity_id)

        return scored

    def _apply_keyword_boost(
        self,
        scored: list[tuple[CatalogItem, float]],
        skills: list[str],
    ) -> list[tuple[CatalogItem, float]]:
        """
        Add a keyword-match bonus to each item's cosine similarity score, then
        re-sort by the combined score.

        Bonus scale (additive, stacks across matched skills):
          +0.20  skill string found verbatim in item name
          +0.08  skill string found verbatim in item description

        Calibrated so a single exact name-match can reorder a 0.50 item above
        a 0.65 item, but cannot override a genuinely strong 0.75+ cosine hit.
        """
        if not skills:
            return scored

        skill_terms = [s.lower() for s in skills]

        boosted: list[tuple[CatalogItem, float]] = []
        for item, base_score in scored:
            bonus = 0.0
            name_lower = item.name.lower()
            desc_lower = item.description.lower()
            for term in skill_terms:
                if term in name_lower:
                    bonus += _KW_BONUS_NAME
                elif term in desc_lower:
                    bonus += _KW_BONUS_DESC
            boosted.append((item, base_score + bonus))

        boosted.sort(key=lambda pair: pair[1], reverse=True)
        return boosted

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clarification_turns_spent(
        self, messages: list[Message], c: TurnClassification
    ) -> int:
        """
        Count how many clarification turns have already been spent in this
        conversation.

        A clarification turn is an assistant turn that happened before any
        shortlist was produced.  We detect this by checking whether the
        classifier found an existing shortlist: if current_shortlist is still
        empty, every prior assistant message was a clarifying question.
        """
        if c.current_shortlist:
            # Recommendations have already been made — we are in refinement,
            # not clarification.  Reset the counter.
            return 0
        return sum(1 for m in messages if m.role == "assistant")

    def _rebuild_shortlist(self, names: list[str]) -> list[Recommendation]:
        result = []
        for name in names:
            item = fuzzy_lookup(self._store, name, threshold=70)
            if item:
                result.append(
                    Recommendation(
                        name=item.name,
                        url=item.url,
                        test_type=item.test_type,
                        keys=item.keys_raw,
                        duration=item.duration_display,
                        languages=item.languages,
                    )
                )
            else:
                logger.warning("Could not resolve shortlist item: %s", name)
        return result

    def _missing_dimension(self, c: TurnClassification) -> str:
        if not c.role_context:
            return "role or job function"
        if not c.seniority and not c.purpose:
            return "seniority level"
        if not c.seniority:
            return "seniority level"
        if not c.purpose:
            return "assessment purpose (selection, development, or screening)"
        # has_enough_context should be True before we reach here;
        # fall back to seniority rather than inventing untracked questions.
        return "seniority level"

    def _assert_no_hallucinated_urls(self, recs: list[Recommendation]) -> None:
        for rec in recs:
            if rec.url not in self._store.valid_urls:
                raise ValueError(f"URL not in catalog: {rec.url}")

    def _attach_shortlist_footer(
        self, reply: str, recommendations: list[Recommendation]
    ) -> str:
        if not recommendations:
            return reply
        names = "; ".join(rec.name for rec in recommendations)
        return f"{reply}\n\nCurrent shortlist: {names}."
