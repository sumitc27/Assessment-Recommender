"""
FastAPI application entry point.

Wires together the catalog store and whichever agent engine is selected via
the RAG_ENGINE environment variable.  All domain logic lives in raw_agent.py
or langchain_agent.py — this file is intentionally thin.
"""

import os

from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI

from app.data_loader import build_store, load_catalog
from app.models import ChatRequest, ChatResponse

load_dotenv()


def _make_agent(store, openai_client):
    engine = os.getenv("RAG_ENGINE", "raw").lower()
    if engine == "langchain":
        from app.agents.langchain_agent import LangchainAgentService
        return LangchainAgentService(store, openai_client)
    from app.agents.raw_agent import RawAgentService
    return RawAgentService(store, openai_client)


def _find_catalog() -> str:
    """Locate the catalog JSON regardless of whether it sits at root or in others/."""
    for candidate in ["shl_product_catalog.json", "others/shl_product_catalog.json"]:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "shl_product_catalog.json not found. "
        "Place it at the project root or in the others/ directory."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    catalog = load_catalog(_find_catalog())
    store = build_store(catalog, openai_client)
    app.state.agent = _make_agent(store, openai_client)
    yield
    # nothing to clean up — store is in-memory


app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not req.messages:
        raise HTTPException(status_code=422, detail="messages list cannot be empty")
    try:
        return app.state.agent.process_turn(req.messages)
    except Exception as exc:
        # Surface the error message without leaking internal traces to the caller.
        # The raw exception is still visible in server logs.
        raise HTTPException(status_code=500, detail=str(exc)) from exc
