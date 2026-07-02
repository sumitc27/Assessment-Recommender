"""
All Pydantic schemas for the SHL recommender.

Three layers:
  - CatalogItem       internal representation of a catalog entry
  - ChatRequest /
    ChatResponse      the external API wire contract (must match the spec exactly)
  - TurnClassification  internal output of the turn classifier LLM call
"""

from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# keys → test_type code mapping
# ---------------------------------------------------------------------------

KEY_TO_CODE: dict[str, str] = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


# ---------------------------------------------------------------------------
# Internal catalog representation
# ---------------------------------------------------------------------------

class CatalogItem(BaseModel):
    """One product from shl_product_catalog.json, with parsed/derived fields added."""

    entity_id: str
    name: str
    url: str                          # from `link`; this is the hallucination guardrail
    description: str
    job_levels: list[str]
    duration_minutes: Optional[int]   # None if empty or "Untimed"
    duration_display: str             # "Untimed", "25 minutes", "" — preserved as-is
    languages: list[str]
    remote: bool
    adaptive: bool
    test_type: str                    # comma-joined codes, e.g. "K" or "A,S"
    keys_raw: list[str]              # original keys array from JSON


# ---------------------------------------------------------------------------
# External API contract (wire schema — must match the assignment spec exactly)
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class Recommendation(BaseModel):
    name: str
    url: str       # must be a link that exists in the catalog
    test_type: str


class ChatResponse(BaseModel):
    # strict=True prevents silent coercion of string "true" → bool, which would
    # mask a bug where the LLM returns the wrong type for end_of_conversation.
    model_config = ConfigDict(strict=True)

    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool


# ---------------------------------------------------------------------------
# Internal turn classifier output
# ---------------------------------------------------------------------------

class TurnClassification(BaseModel):
    """
    Output of the first LLM call (gpt-4o-mini, strict JSON schema).

    The classifier reads the full message history and re-derives all cumulative
    state from scratch — it never relies on a previous classification or any
    server-side session store.
    """

    turn_type: Literal[
        "new_info",
        "refine_add",
        "refine_remove",
        "refine_disambiguate",
        "compare_request",
        "closing_confirm",
        "off_topic_refusal",
    ]

    # Cumulative persona context re-derived from all user turns
    role_context: str     # e.g. "graduate financial analyst"
    seniority: str        # e.g. "Entry-Level", "Senior IC", "" if unknown
    skills: list[str]     # explicit tools/skills named across history
    locale: str           # e.g. "English (USA)", "" if not stated
    purpose: str          # "selection" | "development" | "screening" | ""

    # Action signals from the latest user turn
    named_removals: list[str]    # product names user asked to drop
    compare_targets: list[str]   # exactly two names when turn_type == compare_request
    explicit_adds: list[str]     # product names user explicitly asked to add

    # Re-derived from the most recent assistant turn that had recommendations
    current_shortlist: list[str]

    # Gate for the clarify branch
    has_enough_context: bool     # False → ask ONE clarifying question