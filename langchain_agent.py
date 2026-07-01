"""
LangChain implementation of the assessment recommender agent.

Phase 2 stub — full implementation is Phase 4.
"""

from models import ChatResponse, Message
from data_loader import CatalogStore


class LangchainAgentService:
    def __init__(self, store: CatalogStore, openai_client):
        self._store = store
        self._client = openai_client

    def process_turn(self, messages: list[Message]) -> ChatResponse:
        # Phase 4 stub
        return ChatResponse(
            reply="LangChain engine stub — not yet implemented.",
            recommendations=[],
            end_of_conversation=False,
        )
