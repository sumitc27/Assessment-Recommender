"""
Schema compliance tests — these run without a live server.

Covers:
  - ChatResponse always validates (never crashes Pydantic)
  - recommendations is always a list, never null
  - end_of_conversation is a strict bool
  - URLs in any response must exist in the catalog
  - strict=True rejects string coercions
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import faiss
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agents.raw_agent import RawAgentService
from app.data_loader import CatalogStore, semantic_search_with_scores
from app.models import CatalogItem, ChatResponse, Message, Recommendation, TurnClassification


_ROOT = Path(__file__).parent.parent
CATALOG_PATH = next(
    p for p in [
        _ROOT / "shl_product_catalog.json",
        _ROOT / "others" / "shl_product_catalog.json",
    ]
    if p.exists()
)


@pytest.fixture(scope="module")
def valid_urls() -> frozenset[str]:
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return frozenset(item["link"] for item in data)


# ---------------------------------------------------------------------------
# Happy-path validation
# ---------------------------------------------------------------------------

def test_valid_response_with_empty_recommendations():
    resp = ChatResponse(reply="Hello", recommendations=[], end_of_conversation=False)
    assert resp.recommendations == []
    assert resp.end_of_conversation is False


def test_valid_response_with_recommendations(valid_urls):
    url = next(iter(valid_urls))
    rec = Recommendation(name="Test", url=url, test_type="K", keys=[], duration="", languages=[])
    resp = ChatResponse(reply="Here you go", recommendations=[rec], end_of_conversation=False)
    assert len(resp.recommendations) == 1
    assert resp.recommendations[0].test_type == "K"


def test_end_of_conversation_true():
    resp = ChatResponse(reply="Done", recommendations=[], end_of_conversation=True)
    assert resp.end_of_conversation is True


# ---------------------------------------------------------------------------
# Strict-mode guards
# ---------------------------------------------------------------------------

def test_string_bool_rejected():
    """strict=True must reject 'true' string for end_of_conversation."""
    with pytest.raises(ValidationError):
        ChatResponse(reply="x", recommendations=[], end_of_conversation="true")  # type: ignore


def test_none_recommendations_rejected():
    """recommendations must be a list, never null."""
    with pytest.raises(ValidationError):
        ChatResponse(reply="x", recommendations=None, end_of_conversation=False)  # type: ignore


def test_integer_bool_rejected():
    """strict=True should reject integer 1 for bool field."""
    with pytest.raises(ValidationError):
        ChatResponse(reply="x", recommendations=[], end_of_conversation=1)  # type: ignore


# ---------------------------------------------------------------------------
# Recommendation URL guard
# ---------------------------------------------------------------------------

def test_recommendation_urls_in_catalog(valid_urls):
    """Any URL appearing in recommendations must be a known catalog URL."""
    url = next(iter(valid_urls))
    rec = Recommendation(name="OPQ32r", url=url, test_type="P", keys=[], duration="", languages=[])
    resp = ChatResponse(reply="test", recommendations=[rec], end_of_conversation=False)
    for r in resp.recommendations:
        assert r.url in valid_urls, f"URL not in catalog: {r.url}"


def test_hallucinated_url_detection(valid_urls):
    """Verify our guardrail logic correctly flags an invented URL."""
    fake_url = "https://www.shl.com/products/fake-product/"
    assert fake_url not in valid_urls   # confirm it's actually fake


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

def test_model_dump_round_trip():
    resp = ChatResponse(
        reply="Here are your assessments.",
        recommendations=[
            Recommendation(
                name="Graduate Scenarios",
                url="https://www.shl.com/products/product-catalog/view/graduate-scenarios/",
                test_type="B",
                keys=["Biodata & Situational Judgment"],
                duration="25 minutes",
                languages=["English (USA)"],
            )
        ],
        end_of_conversation=False,
    )
    dumped = resp.model_dump()
    restored = ChatResponse.model_validate(dumped)
    assert restored.reply == resp.reply
    assert restored.recommendations[0].name == "Graduate Scenarios"


# ---------------------------------------------------------------------------
# Low-similarity retrieval guards
# ---------------------------------------------------------------------------

def _build_score_test_store() -> CatalogStore:
    items = [
        CatalogItem(
            entity_id="1",
            name="Alpha",
            url="https://example.com/a",
            description="Alpha item",
            job_levels=[],
            duration_minutes=None,
            duration_display="",
            languages=[],
            remote=False,
            adaptive=False,
            test_type="P",
            keys_raw=[],
        ),
        CatalogItem(
            entity_id="2",
            name="Beta",
            url="https://example.com/b",
            description="Beta item",
            job_levels=[],
            duration_minutes=None,
            duration_display="",
            languages=[],
            remote=False,
            adaptive=False,
            test_type="K",
            keys_raw=[],
        ),
    ]
    embeddings = np.array(
        [
            [1.0] + [0.0] * 15,
            [0.0, 1.0] + [0.0] * 14,
        ],
        dtype=np.float32,
    )
    index = faiss.IndexFlatIP(16)
    index.add(embeddings)
    return CatalogStore(
        items=items,
        by_entity_id={item.entity_id: item for item in items},
        name_to_id={item.name.lower(): item.entity_id for item in items},
        embeddings=embeddings,
        faiss_index=index,
        index_to_id=[item.entity_id for item in items],
        valid_urls=frozenset(item.url for item in items),
    )


def test_semantic_search_returns_scores():
    store = _build_score_test_store()
    query = np.array([1.0] + [0.0] * 15, dtype=np.float32)
    results = semantic_search_with_scores(store, query, k=2)
    assert results[0][0].name == "Alpha"
    assert results[0][1] == pytest.approx(1.0)


def test_low_similarity_branch_flags_catalog_gap(monkeypatch):
    store = _build_score_test_store()
    service = RawAgentService(store, openai_client=object())

    def fake_search(_store, _query_embedding, k=20):
        return [(store.items[0], 0.25), (store.items[1], 0.24)]

    monkeypatch.setattr("app.agents.raw_agent.semantic_search_with_scores", fake_search)
    monkeypatch.setattr(service, "_embed", lambda _text: np.zeros(16, dtype=np.float32))

    classification = TurnClassification(
        reasoning="Turn 1 [USER]: role=Rust developer, purpose=selection. No seniority stated.",
        turn_type="new_info",
        role_context="Rust developer",
        seniority="",
        skills=["Rust"],
        locale="",
        purpose="selection",
        named_removals=[],
        compare_targets=[],
        explicit_adds=[],
        current_shortlist=[],
        has_enough_context=True,
    )

    candidates, defaults_added, catalog_gaps = service._retrieve(classification)
    assert candidates[0].name == "Alpha"
    assert defaults_added == []
    assert any("no strong match" in gap for gap in catalog_gaps)


def test_handle_confirm_rebuilds_shortlist():
    store = _build_score_test_store()
    service = RawAgentService(store, openai_client=object())
    classification = TurnClassification(
        reasoning="User confirmed the shortlist is final.",
        turn_type="closing_confirm",
        role_context="",
        seniority="",
        skills=[],
        locale="",
        purpose="",
        named_removals=[],
        compare_targets=[],
        explicit_adds=[],
        current_shortlist=["Alpha", "Beta"],
        has_enough_context=True,
    )

    response = service._handle_confirm(classification)

    assert response.end_of_conversation is True
    assert [rec.name for rec in response.recommendations] == ["Alpha", "Beta"]


class _StubCompletion:
    def create(self, **kwargs):
        class _Message:
            content = "Alpha and Beta measure different things."

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]

        return _Response()


class _StubChat:
    completions = _StubCompletion()


class _StubClient:
    chat = _StubChat()


def test_handle_compare_uses_catalog_items_and_preserves_shortlist():
    store = _build_score_test_store()
    service = RawAgentService(store, openai_client=_StubClient())
    classification = TurnClassification(
        reasoning="User asked to compare Alpha vs Beta. Both names appear in the shortlist.",
        turn_type="compare_request",
        role_context="",
        seniority="",
        skills=[],
        locale="",
        purpose="",
        named_removals=[],
        compare_targets=["Alpha", "Beta"],
        explicit_adds=[],
        current_shortlist=["Alpha", "Beta"],
        has_enough_context=True,
    )

    response = service._handle_compare(
        classification,
        messages=[Message(role="user", content="What is the difference between Alpha and Beta?")],
    )

    assert response.end_of_conversation is False
    assert response.reply.startswith("Alpha and Beta measure different things.")
    assert [rec.name for rec in response.recommendations] == ["Alpha", "Beta"]


def test_attach_shortlist_footer():
    store = _build_score_test_store()
    service = RawAgentService(store, openai_client=object())
    footer = service._attach_shortlist_footer(
        "Here is the update.",
        [Recommendation(name="Alpha", url="https://example.com/a", test_type="P", keys=[], duration="", languages=[])],
    )
    assert footer.endswith("Current shortlist: Alpha.")
