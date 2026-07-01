"""
OpenAI SDK implementation of the assessment recommender agent.

Phase 2 stub: process_turn returns a hardcoded valid ChatResponse so the
FastAPI shell can be verified before any LLM logic is wired in.
Full implementation is Phase 3.
"""

from models import ChatResponse, Message
from data_loader import CatalogStore


class RawAgentService:
    def __init__(self, store: CatalogStore, openai_client):
        self._store = store
        self._client = openai_client

    def process_turn(self, messages: list[Message]) -> ChatResponse:
        # Phase 2 stub — replaced in Phase 3
        return ChatResponse(
            reply="I can help you find the right SHL assessments. Could you tell me the role you're hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )
