# SHL Assessment Recommender

A conversational AI service that recommends SHL assessments based on a hiring persona. Built as a stateless FastAPI application — every `/chat` call carries the full conversation history, and all state is re-derived on each turn with no server-side session store.

---

## Architecture Overview

```
POST /chat  →  Turn Classifier (gpt-4o-mini)
                    │
         ┌──────────┴──────────────────────────┐
         │                                     │
    has_enough_context?                   turn_type?
         │                                     │
       No → Clarify (≤2 turns)        compare_request → Compare LLM
       Yes → Retrieve → Compose        closing_confirm → Rebuild shortlist
                                       off_topic       → Refusal template
```

Two LLM calls maximum per turn:
1. **Classifier** — structured JSON output, re-derives all cumulative state from full message history
2. **Composer / Compare** — free-text reply grounded in retrieved catalog items

Two swappable engines behind a single interface, toggled via `RAG_ENGINE` env var:

| Engine | `RAG_ENGINE` | LLM calls | Retrieval |
|--------|-------------|-----------|-----------|
| Raw (default) | `raw` | OpenAI SDK, `response_format=json_schema` | Direct FAISS |
| LangChain | `langchain` | `.with_structured_output()` | Delegates to raw engine |

---

## Retrieval Pipeline

The 377-item catalog is embedded once at startup (`text-embedding-3-small`, L2-normalized, FAISS `IndexFlatIP`). Each item is embedded as `"{name}. {description}. Levels: {job_levels}"`.

Per turn, retrieval runs three stages:

1. **Dense FAISS search** (k=20) against a persona query built from role + seniority + skills + purpose + locale
2. **Keyword injection** — linear scan of all 377 items for exact skill-keyword matches the embedding search missed (catches niche acronyms and framework names)
3. **Keyword boost + re-sort** — items whose name (+0.20) or description (+0.08) contains stated skill terms are promoted, then re-sorted by combined score

A **hard relevance gate at 0.50** discards anything below that threshold before the composer sees it. A **default-bundling step** then adds a role-appropriate personality measure if none survived:

- Safety / industrial roles → DSI or Manufacturing Safety bundle
- Sales roles → OPQ MQ Sales Report
- Contact centre roles → Entry Level Customer Serv bundle  
- All others → OPQ32r (with explicit opt-out offered in the reply)

Final list is capped at **5 items**.

---

## Project Structure

```
├── main.py                          # FastAPI app — /health + /chat
├── app/
│   ├── models.py                    # All Pydantic schemas (CatalogItem, ChatRequest/Response, TurnClassification)
│   ├── data_loader.py               # Catalog loading, FAISS index, fuzzy lookup
│   ├── prompts.py                   # All LLM prompt templates
│   └── agents/
│       ├── raw_agent.py             # OpenAI SDK engine (primary)
│       └── langchain_agent.py       # LangChain engine (alternative)
├── evaluate.py                      # Trace replay evaluation harness
├── tests/
│   ├── test_schema.py               # Schema compliance + retrieval unit tests (no server needed)
│   ├── test_traces.py               # Full trace replay tests
│   └── test_adversarial.py          # Behavioral probes (premature rec, removal, refusal, hallucination)
├── GenAI_SampleConversations/       # 10 labeled conversation traces (C1–C10)
└── shl_product_catalog.json         # 377-item SHL product catalog
```

---

## Setup

### Prerequisites

- Python 3.10+
- An OpenAI API key with access to `gpt-4o-mini` and `text-embedding-3-small`

### Install

```bash
pip install -r requirements.txt
```

### Environment

Create a `.env` file at the project root:

```env
OPENAI_API_KEY=sk-...
RAG_ENGINE=raw          # or "langchain"
```

---

## Running the Server

```bash
uvicorn main:app --reload
```

The server embeds the full 377-item catalog on startup (single OpenAI embeddings call, ~5 seconds). After that:

```bash
# Health check
curl http://localhost:8000/health
# → {"status": "ok"}
```

---

## API

### `POST /chat`

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I need to hire a mid-level Java developer"},
    {"role": "assistant", "content": "What is the assessment purpose — selection, development, or screening?"},
    {"role": "user", "content": "Selection"}
  ]
}
```

**Response:**
```json
{
  "reply": "For a mid-level Java developer focused on selection, here are five assessments...",
  "recommendations": [
    {
      "name": "Verify - Numerical Reasoning",
      "url": "https://www.shl.com/products/product-catalog/view/verify-numerical-reasoning/",
      "test_type": "A",
      "keys": ["Ability & Aptitude"],
      "duration": "17 minutes",
      "languages": ["English (USA)", "French", "German"],
      "score": 0.7132
    }
  ],
  "end_of_conversation": false
}
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `reply` | string | Natural-language response from the agent |
| `recommendations` | array | Up to 5 catalog-grounded assessments (empty during clarification) |
| `recommendations[].score` | float \| null | Cosine + keyword boost score (0–1); null for shortlist rebuilds |
| `end_of_conversation` | bool | `true` only on explicit user confirmation |

The full conversation history must be sent on every call — the server holds no session state.

---

## Key Design Decisions

**Stateless by design.** The turn classifier reads the entire message history and re-derives role, seniority, skills, removals, and the current shortlist from scratch on every call. This makes the service trivially horizontally scalable and eliminates a class of stale-state bugs.

**Chain-of-Thought scratchpad.** The classifier's JSON schema places a `reasoning` field first, forcing gpt-4o-mini to write a turn-by-turn edit ledger before committing to any extracted value. This fixed contradictory-edit drift (e.g. add skill → remove skill → add again).

**2-turn soft clarification budget.** After two clarifying assistant turns, the agent recommends with a stated caveat rather than asking a third question, preventing infinite clarification loops.

**Hallucination guardrail.** All URLs in responses are assembled from `store.by_entity_id` in agent code — never extracted from LLM output. A post-build assertion validates every URL against the catalog's link set before the response is returned.

---

## Running Tests

```bash
# Unit tests — no server or API key required
pytest tests/test_schema.py -v

# Full trace replay — requires live server on localhost:8000
pytest tests/test_traces.py -v

# Behavioral probes — requires live server
pytest tests/test_adversarial.py -v

# All tests
pytest tests/ -v --tb=short
```

---

## Evaluation

```bash
# Replay all 10 labeled traces and report Recall@5 + schema compliance
python evaluate.py

# Against a custom server URL
python evaluate.py --url http://your-server:8000
```

The harness checks three things per trace turn:
1. `ChatResponse` Pydantic validation passes
2. All recommendation URLs exist in the catalog
3. Recall@5 on the final recommendation turn against expected items from the trace

---

## Switching Engines

```bash
# LangChain engine
RAG_ENGINE=langchain uvicorn main:app --port 8001
```

Both engines produce identical API responses. The LangChain engine uses `.with_structured_output()` for classification and LCEL pipes for composition, but delegates all retrieval and bundling logic to the raw engine to avoid duplication.

---

## Built With

- [FastAPI](https://fastapi.tiangolo.com/) — API framework
- [OpenAI Python SDK](https://github.com/openai/openai-python) — LLM + embeddings
- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search
- [LangChain](https://python.langchain.com/) — Alternative engine wiring
- [Pydantic v2](https://docs.pydantic.dev/) — Schema validation
- [rapidfuzz](https://github.com/maxbachmann/RapidFuzz) — Fuzzy name matching for compare/remove paths
