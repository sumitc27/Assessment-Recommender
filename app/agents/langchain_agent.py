"""
LangChain implementation of the assessment recommender agent.

Uses the same routing logic and prompt templates as RawAgentService but swaps
the SDK layer: structured output via .with_structured_output(), retrieval via
a LangChain FAISS vectorstore, composition via LCEL pipes.

Both engines expose the same interface:
    def process_turn(self, messages: list[Message]) -> ChatResponse
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS as LangchainFAISS

from app.data_loader import CatalogStore, fuzzy_lookup
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
# Reuse the shared retrieval helpers from the raw agent
from app.agents.raw_agent import RawAgentService

logger = logging.getLogger(__name__)

_OPQ32R_NAME = "Occupational Personality Questionnaire OPQ32r"


class LangchainAgentService:
    """
    Same behaviour as RawAgentService; different SDK wiring.

    Differences vs raw:
      - Classifier uses .with_structured_output() instead of json_schema response_format
      - Retrieval uses a LangChain FAISS vectorstore (built from the pre-loaded catalog)
      - Composition uses LCEL: prompt | llm | StrOutputParser()

    The routing logic, default-bundling, removal handling, and refusal behaviour
    are shared by delegating to a RawAgentService instance for those helpers.
    """

    def __init__(self, store: CatalogStore, openai_client):
        self._store = store

        self._llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self._classifier = self._llm.with_structured_output(TurnClassification)

        # Build a LangChain FAISS vectorstore from the pre-loaded catalog
        self._vectorstore = self._build_vectorstore(store)

        # Reuse non-LLM helpers from RawAgentService (retrieval, bundling, etc.)
        self._raw = RawAgentService(store, openai_client)

    # ------------------------------------------------------------------
    # Public entry point (same signature as RawAgentService)
    # ------------------------------------------------------------------

    def process_turn(self, messages: list[Message]) -> ChatResponse:
        classification = self._classify(messages)
        turn = classification.turn_type

        if turn == "off_topic_refusal":
            return self._raw._handle_refusal(classification)
        if turn == "closing_confirm":
            return self._raw._handle_confirm(classification)
        if turn == "compare_request":
            return self._handle_compare(classification, messages)
        if not classification.has_enough_context:
            return self._handle_clarify(classification)

        if turn in {"new_info", "refine_add", "refine_remove", "refine_disambiguate"}:
            return self._handle_refine(classification)

        return self._handle_recommend(classification)

    # ------------------------------------------------------------------
    # Classifier — uses .with_structured_output() instead of json_schema
    # ------------------------------------------------------------------

    def _classify(self, messages: list[Message]) -> TurnClassification:
        history_text = "\n".join(
            f"[{m.role.upper()}]: {m.content}" for m in messages
        )
        prompt_text = (
            CLASSIFIER_SYSTEM
            + "\n\n"
            + CLASSIFIER_USER_TEMPLATE.format(history=history_text)
        )
        return self._classifier.invoke(prompt_text)

    # ------------------------------------------------------------------
    # Compare — LCEL pipe
    # ------------------------------------------------------------------

    def _handle_compare(
        self, c: TurnClassification, messages: list[Message]
    ) -> ChatResponse:
        targets = c.compare_targets
        if len(targets) < 2:
            return ChatResponse(
                reply="I can compare two named assessments once you tell me both product names.",
                recommendations=self._raw._rebuild_shortlist(c.current_shortlist),
                end_of_conversation=False,
            )

        item_a = fuzzy_lookup(self._store, targets[0])
        item_b = fuzzy_lookup(self._store, targets[1])

        if item_a is None or item_b is None:
            missing = targets[0] if item_a is None else targets[1]
            return ChatResponse(
                reply=f'I couldn\'t find "{missing}" in the SHL catalog.',
                recommendations=self._raw._rebuild_shortlist(c.current_shortlist),
                end_of_conversation=False,
            )

        prompt = ChatPromptTemplate.from_messages([
            ("system", COMPARE_SYSTEM),
            ("human", COMPARE_USER_TEMPLATE),
        ])
        chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0.2) | StrOutputParser()
        reply = chain.invoke({
            "question": messages[-1].content,
            "name_a": item_a.name,
            "desc_a": item_a.description,
            "name_b": item_b.name,
            "desc_b": item_b.description,
        })
        return ChatResponse(
            reply=reply.strip(),
            recommendations=self._raw._rebuild_shortlist(c.current_shortlist),
            end_of_conversation=False,
        )

    # ------------------------------------------------------------------
    # Clarify — LCEL pipe
    # ------------------------------------------------------------------

    def _handle_clarify(self, c: TurnClassification) -> ChatResponse:
        missing = self._raw._missing_dimension(c)
        prompt = ChatPromptTemplate.from_messages([
            ("system", CLARIFY_SYSTEM),
            ("human", CLARIFY_USER_TEMPLATE),
        ])
        chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0.3) | StrOutputParser()
        question = chain.invoke({
            "role_context": c.role_context or "(unknown)",
            "seniority": c.seniority or "(unknown)",
            "purpose": c.purpose or "(unknown)",
            "locale": c.locale or "(not stated)",
            "missing": missing,
        })
        return ChatResponse(
            reply=question.strip(),
            recommendations=[],
            end_of_conversation=False,
        )

    # ------------------------------------------------------------------
    # Recommend — uses LangChain vectorstore retrieval + LCEL compose
    # ------------------------------------------------------------------

    def _handle_recommend(self, c: TurnClassification) -> ChatResponse:
        persona_query = self._raw._build_persona_query(c)

        # Retrieve via LangChain FAISS
        docs_and_scores = self._vectorstore.similarity_search_with_relevance_scores(persona_query, k=20)
        candidates = [
            self._store.by_entity_id[doc.metadata["entity_id"]]
            for doc, _score in docs_and_scores
            if doc.metadata.get("entity_id") in self._store.by_entity_id
        ]

        # Shared post-retrieval logic (rerank, remove, bundle)
        candidates = self._raw._rerank(candidates, c)
        candidates = self._raw._apply_removals(candidates, c.named_removals)

        catalog_gaps: list[str] = []
        top_score = docs_and_scores[0][1] if docs_and_scores else None
        if top_score is not None and top_score < 0.28:
            substitute = candidates[0].name if candidates else None
            signal = c.skills[0] if c.skills else (c.role_context or "this request")
            gap_note = f'no strong match for "{signal}" in the catalog'
            if substitute:
                gap_note += f"; using {substitute} as the closest substitute"
            catalog_gaps.append(gap_note)

        for add_name in c.explicit_adds:
            match = fuzzy_lookup(self._store, add_name)
            if match and match not in candidates:
                candidates.insert(0, match)
            elif match is None:
                catalog_gaps.append(add_name)

        candidates, defaults_added = self._raw._apply_default_bundling(candidates, c)
        if not catalog_gaps:
            catalog_gaps = self._raw._detect_skill_gaps(c.skills, candidates)

        candidates = candidates[:10]

        shortlist_text = "\n".join(
            f"  {i+1}. {item.name} [{item.test_type}] — {item.description[:100]}..."
            for i, item in enumerate(candidates)
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", COMPOSER_SYSTEM),
            ("human", COMPOSER_USER_TEMPLATE),
        ])
        chain = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0.4) | StrOutputParser()
        reply = chain.invoke({
            "role_context": c.role_context or "unspecified",
            "seniority": c.seniority or "unspecified",
            "skills": ", ".join(c.skills) if c.skills else "none specified",
            "purpose": c.purpose or "unspecified",
            "locale": c.locale or "not specified",
            "count": len(candidates),
            "shortlist_text": shortlist_text,
            "defaults_added": ", ".join(defaults_added) if defaults_added else "none",
            "catalog_gaps": ", ".join(catalog_gaps) if catalog_gaps else "none",
        })

        recommendations = [
            Recommendation(name=item.name, url=item.url, test_type=item.test_type)
            for item in candidates
        ]
        self._raw._assert_no_hallucinated_urls(recommendations)

        return ChatResponse(
            reply=reply.strip(),
            recommendations=recommendations,
            end_of_conversation=False,
        )

    def _handle_refine(self, c: TurnClassification) -> ChatResponse:
        """Refine turns currently reuse the recommendation pipeline."""
        return self._handle_recommend(c)

    # ------------------------------------------------------------------
    # Vectorstore builder
    # ------------------------------------------------------------------

    def _build_vectorstore(self, store: CatalogStore) -> LangchainFAISS:
        """
        Build a LangChain FAISS vectorstore from the pre-loaded catalog.

        Each document's page_content is the same embedding text used by the raw
        engine, and entity_id is stored in metadata for lookup after retrieval.
        """
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
        documents = [
            Document(
                page_content=(
                    f"{item.name}. {item.description} "
                    f"Levels: {', '.join(item.job_levels) or 'General'}"
                ),
                metadata={
                    "entity_id": item.entity_id,
                    "test_type": item.test_type,
                    "url": item.url,
                },
            )
            for item in store.items
        ]
        return LangchainFAISS.from_documents(documents, embeddings_model)
