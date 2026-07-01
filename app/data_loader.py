"""
Catalog ingestion, FAISS index construction, and name-based item lookup.

The CatalogStore is built once at application startup and injected into both
agent implementations. Nothing here makes LLM calls when running without an
OpenAI client (useful for unit tests against catalog loading logic).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from rapidfuzz import process as fuzz_process

from app.models import KEY_TO_CODE, CatalogItem


# ---------------------------------------------------------------------------
# Duration parsing
# ---------------------------------------------------------------------------

def _parse_duration(raw: str) -> tuple[Optional[int], str]:
    """
    Return (minutes_as_int_or_None, display_string).

    "Untimed" is a real, distinct SHL value — not missing data.  It gets
    duration_minutes=None but a non-empty display string so callers can
    distinguish it from a genuinely missing value.
    """
    stripped = raw.strip()
    if not stripped:
        return None, ""
    if stripped.lower() == "untimed":
        return None, "Untimed"
    match = re.search(r"(\d+)", stripped)
    if match:
        return int(match.group(1)), stripped
    return None, stripped


# ---------------------------------------------------------------------------
# test_type derivation
# ---------------------------------------------------------------------------

def _derive_test_type(keys: list[str]) -> str:
    """
    Map the catalog 'keys' array to a comma-joined code string.

    Unknown key labels are skipped rather than raising — the catalog may gain
    new categories after this code was written.
    """
    codes = [KEY_TO_CODE[k] for k in keys if k in KEY_TO_CODE]
    return ",".join(codes) if codes else ""


# ---------------------------------------------------------------------------
# JSON ingestion
# ---------------------------------------------------------------------------

def load_catalog(path: str | Path) -> list[CatalogItem]:
    """Load and parse shl_product_catalog.json into CatalogItem objects."""
    with open(path, encoding="utf-8") as fh:
        raw_items: list[dict] = json.load(fh)

    items: list[CatalogItem] = []
    for entry in raw_items:
        duration_minutes, duration_display = _parse_duration(
            entry.get("duration_raw") or entry.get("duration") or ""
        )
        items.append(
            CatalogItem(
                entity_id=entry["entity_id"],
                name=entry["name"],
                url=entry["link"],
                description=entry.get("description") or "",
                job_levels=entry.get("job_levels") or [],
                duration_minutes=duration_minutes,
                duration_display=duration_display,
                languages=entry.get("languages") or [],
                remote=entry.get("remote", "no").lower() == "yes",
                adaptive=entry.get("adaptive", "no").lower() == "yes",
                test_type=_derive_test_type(entry.get("keys") or []),
                keys_raw=entry.get("keys") or [],
            )
        )
    return items


# ---------------------------------------------------------------------------
# Embedding text builder
# ---------------------------------------------------------------------------

def _embedding_text(item: CatalogItem) -> str:
    """
    Build the text that gets embedded for semantic retrieval.

    Including job_levels as text helps the embedding space differentiate
    entry-level from executive products when the user's query names a level.
    """
    levels = ", ".join(item.job_levels) if item.job_levels else "General"
    return f"{item.name}. {item.description} Levels: {levels}"


# ---------------------------------------------------------------------------
# CatalogStore
# ---------------------------------------------------------------------------

@dataclass
class CatalogStore:
    items: list[CatalogItem]
    by_entity_id: dict[str, CatalogItem]
    # lowercased item name → entity_id (for O(1) exact lookup)
    name_to_id: dict[str, str]
    # parallel arrays: embeddings[i] corresponds to index_to_id[i]
    embeddings: np.ndarray       # shape (N, dim), float32, L2-normalised
    faiss_index: faiss.IndexFlatIP
    index_to_id: list[str]
    # set of all valid URLs — used as the hallucination guardrail
    valid_urls: frozenset[str] = field(default_factory=frozenset)


def build_store(catalog: list[CatalogItem], openai_client) -> CatalogStore:
    """
    Build the in-memory store from a loaded catalog.

    Requires an OpenAI client only for the embedding step.  The rest of the
    store (lookup dicts, name index) is built from the catalog alone.

    Called once at startup; the returned store is shared across all requests.
    """
    by_entity_id = {item.entity_id: item for item in catalog}
    name_to_id = {item.name.lower(): item.entity_id for item in catalog}

    # Embed all items in a single API call (377 items is well within the limit)
    texts = [_embedding_text(item) for item in catalog]
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    # API returns embeddings in the same order as the input list
    raw_vecs = np.array(
        [e.embedding for e in response.data], dtype=np.float32
    )

    # L2-normalise so that inner product == cosine similarity
    norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)   # guard against zero-norm vectors
    embeddings = raw_vecs / norms

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_to_id = [item.entity_id for item in catalog]
    valid_urls = frozenset(item.url for item in catalog)

    return CatalogStore(
        items=catalog,
        by_entity_id=by_entity_id,
        name_to_id=name_to_id,
        embeddings=embeddings,
        faiss_index=index,
        index_to_id=index_to_id,
        valid_urls=valid_urls,
    )


# ---------------------------------------------------------------------------
# Fuzzy name lookup (used by the compare path and explicit-add handling)
# ---------------------------------------------------------------------------

def fuzzy_lookup(
    store: CatalogStore,
    query: str,
    threshold: int = 80,
) -> Optional[CatalogItem]:
    """
    Return the CatalogItem whose name best matches `query`, or None if the
    best match scores below `threshold`.

    Uses token_sort_ratio so that "OPQ32r" and "Occupational Personality
    Questionnaire OPQ32r" both resolve to the same item.
    """
    all_names = list(store.name_to_id.keys())
    result = fuzz_process.extractOne(
        query.lower(),
        all_names,
        score_cutoff=threshold,
    )
    if result is None:
        return None
    matched_name, _score, _idx = result
    entity_id = store.name_to_id[matched_name]
    return store.by_entity_id[entity_id]


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------

def semantic_search(
    store: CatalogStore,
    query_embedding: np.ndarray,
    k: int = 20,
) -> list[CatalogItem]:
    """
    Return the top-k catalog items by cosine similarity to query_embedding.

    query_embedding must already be L2-normalised (same space as store.embeddings).
    """
    return [item for item, _score in semantic_search_with_scores(store, query_embedding, k)]


def semantic_search_with_scores(
    store: CatalogStore,
    query_embedding: np.ndarray,
    k: int = 20,
) -> list[tuple[CatalogItem, float]]:
    """Return the top-k catalog items together with their cosine similarity."""
    vec = query_embedding.reshape(1, -1).astype(np.float32)
    scores, indices = store.faiss_index.search(vec, k)
    results: list[tuple[CatalogItem, float]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        entity_id = store.index_to_id[idx]
        results.append((store.by_entity_id[entity_id], float(score)))
    return results


# ---------------------------------------------------------------------------
# Standalone sanity check (run with: python data_loader.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from collections import Counter

    catalog = load_catalog("shl_product_catalog.json")
    print(f"Loaded {len(catalog)} catalog items")

    key_counts: Counter[str] = Counter()
    for item in catalog:
        key_counts.update(item.keys_raw)

    print("\nKeys distribution:")
    for key, count in key_counts.most_common():
        code = KEY_TO_CODE.get(key, "?")
        print(f"  [{code}] {key}: {count}")

    no_desc = sum(1 for item in catalog if not item.description)
    no_lang = sum(1 for item in catalog if not item.languages)
    no_dur = sum(1 for item in catalog if item.duration_display == "")
    untimed = sum(1 for item in catalog if item.duration_display == "Untimed")
    print(f"\nItems with no description : {no_desc}")
    print(f"Items with no languages   : {no_lang}")
    print(f"Items with no duration    : {no_dur}")
    print(f"Items marked 'Untimed'    : {untimed}")

    print("\nSample item:")
    sample = next(i for i in catalog if i.test_type == "A,S")
    print(f"  name         : {sample.name}")
    print(f"  test_type    : {sample.test_type}")
    print(f"  job_levels   : {sample.job_levels}")
    print(f"  duration     : {sample.duration_display}")
    print(f"  url          : {sample.url}")
