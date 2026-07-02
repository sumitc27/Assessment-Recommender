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
_CLASSIFIER_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "turn_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
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
                "turn_type", "role_context", "seniority", "skills", "locale",
                "purpose", "named_removals", "compare_targets", "explicit_adds",
                "current_shortlist", "has_enough_context",
            ],
            "additionalProperties": False,
        },
    },
}

_OPQ32R_NAME = "Occupational Personality Questionnaire OPQ32r"
_LOW_SIMILARITY_THRESHOLD = 0.28


class RawAgentService:
    def __init__(self, store: CatalogStore, openai_client):
        self._store = store
        self._client = openai_client

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

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
        reply = REFUSAL_TEMPLATE.format(topic="legal or compliance")
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
            temperature=0.3,
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

    def _handle_recommend(self, c: TurnClassification) -> ChatResponse:
        candidates, defaults_added, catalog_gaps = self._retrieve(c)

        shortlist_text = "\n".join(
            f"  {i+1}. {item.name} [{item.test_type}] — {item.description[:100]}..."
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
            Recommendation(name=item.name, url=item.url, test_type=item.test_type)
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
    ) -> tuple[list[CatalogItem], list[str], list[str]]:
        """Returns (candidates, defaults_added, catalog_gaps).

        For refine turns with an established shortlist the existing items are
        preserved as the starting point so the user never loses their list.
        For new_info (or when there is no prior shortlist) a fresh FAISS search
        is performed.
        """
        catalog_gaps: list[str] = []
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

            # For refine_add: supplement with fresh FAISS results to fill gaps
            if c.turn_type == "refine_add" and len(candidates) < 10:
                query_vec = self._embed(self._build_persona_query(c))
                scored = semantic_search_with_scores(self._store, query_vec, k=20)
                for item, _score in scored:
                    if item not in candidates and len(candidates) < 10:
                        candidates.append(item)

        else:
            # Fresh retrieval for new_info or when no prior shortlist exists
            query_vec = self._embed(self._build_persona_query(c))
            scored_candidates = semantic_search_with_scores(self._store, query_vec, k=20)
            candidates = [item for item, _score in scored_candidates]
            candidates = self._rerank(candidates, c)
            candidates = self._apply_removals(candidates, c.named_removals)

            top_score = scored_candidates[0][1] if scored_candidates else None
            if top_score is not None and top_score < _LOW_SIMILARITY_THRESHOLD:
                substitute = candidates[0].name if candidates else None
                signal = c.skills[0] if c.skills else (c.role_context or "this request")
                gap_note = f'no strong match for "{signal}" in the catalog'
                if substitute:
                    gap_note += f"; using {substitute} as the closest substitute"
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

        return candidates[:10], defaults_added, catalog_gaps

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rebuild_shortlist(self, names: list[str]) -> list[Recommendation]:
        result = []
        for name in names:
            item = fuzzy_lookup(self._store, name, threshold=70)
            if item:
                result.append(
                    Recommendation(name=item.name, url=item.url, test_type=item.test_type)
                )
            else:
                logger.warning("Could not resolve shortlist item: %s", name)
        return result

    def _missing_dimension(self, c: TurnClassification) -> str:
        if not c.role_context:
            return "role"
        if not c.seniority:
            return "seniority level"
        if not c.purpose:
            return "purpose (selection, development, or screening)"
        return "additional context"

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
