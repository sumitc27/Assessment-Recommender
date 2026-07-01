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

from data_loader import build_store, load_catalog
from models import ChatRequest, ChatResponse

load_dotenv()


def _make_agent(store, openai_client):
    engine = os.getenv("RAG_ENGINE", "raw").lower()
    if engine == "langchain":
        from langchain_agent import LangchainAgentService
        return LangchainAgentService(store, openai_client)
    from raw_agent import RawAgentService
    return RawAgentService(store, openai_client)


@asynccontextmanager
async def lifespan(app: FastAPI):
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    catalog = load_catalog("shl_product_catalog.json")
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
