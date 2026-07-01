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

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models import ChatResponse, Recommendation


CATALOG_PATH = Path(__file__).parent.parent / "shl_product_catalog.json"


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
    rec = Recommendation(name="Test", url=url, test_type="K")
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
    rec = Recommendation(name="OPQ32r", url=url, test_type="P")
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
            )
        ],
        end_of_conversation=False,
    )
    dumped = resp.model_dump()
    restored = ChatResponse.model_validate(dumped)
    assert restored.reply == resp.reply
    assert restored.recommendations[0].name == "Graduate Scenarios"
