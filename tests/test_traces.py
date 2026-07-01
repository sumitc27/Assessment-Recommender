"""
Trace replay tests — replay all 10 C*.md conversations turn-by-turn against
a running server and assert Recall@10 on final shortlists.

Requires:
    uvicorn main:app running on localhost:8000  (or TEST_SERVER_URL env var)

Skip automatically if the server is not reachable.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

TRACES_DIR = Path(__file__).parent.parent / "GenAI_SampleConversations"
CATALOG_PATH = Path(__file__).parent.parent / "shl_product_catalog.json"
BASE_URL = os.getenv("TEST_SERVER_URL", "http://localhost:8000")
RECALL_THRESHOLD = 0.6   # at least 60% of expected items must appear


# ---------------------------------------------------------------------------
# Skip entire module if server is unreachable
# ---------------------------------------------------------------------------

def _server_available() -> bool:
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _server_available(),
    reason=f"Server not reachable at {BASE_URL} — start with: uvicorn main:app",
)


# ---------------------------------------------------------------------------
# Trace parser (same logic as evaluate.py, kept here for test isolation)
# ---------------------------------------------------------------------------

@dataclass
class TraceTurn:
    user_message: str
    expected_items: list[str]
    expected_end: bool


def parse_trace(md_path: Path) -> list[TraceTurn]:
    text = md_path.read_text(encoding="utf-8")
    turn_blocks = re.split(r"###\s+Turn\s+\d+", text)[1:]
    turns = []
    for block in turn_blocks:
        user_match = re.search(r"\*\*User\*\*\s*\n+>\s*(.+)", block)
        if not user_match:
            continue
        table_rows = re.findall(r"^\|\s*\d+\s*\|\s*([^|]+)\|", block, re.MULTILINE)
        end_match = re.search(
            r"`end_of_conversation`.*?\*\*(true|false)\*\*", block, re.IGNORECASE
        )
        turns.append(TraceTurn(
            user_message=user_match.group(1).strip(),
            expected_items=[r.strip() for r in table_rows if r.strip()],
            expected_end=end_match.group(1).lower() == "true" if end_match else False,
        ))
    return turns


def recall(got: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0
    got_lower = {n.lower() for n in got}
    hits = sum(1 for e in expected if e.lower() in got_lower)
    return hits / len(expected)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def valid_urls() -> frozenset[str]:
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return frozenset(item["link"] for item in data)


@pytest.fixture(scope="module")
def client() -> httpx.Client:
    with httpx.Client(base_url=BASE_URL, timeout=30) as c:
        yield c


def _trace_ids() -> list[str]:
    return sorted(p.stem for p in TRACES_DIR.glob("C*.md"))


# ---------------------------------------------------------------------------
# Parametrized trace replay
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("trace_id", _trace_ids())
def test_trace_schema_and_recall(trace_id, client, valid_urls):
    """
    For each trace:
      1. Replay all turns in order
      2. Assert every response is schema-valid
      3. Assert no hallucinated URLs
      4. Assert Recall@10 >= threshold on turns with expected items
      5. Assert end_of_conversation is correct on the final turn
    """
    md_path = TRACES_DIR / f"{trace_id}.md"
    turns = parse_trace(md_path)
    assert turns, f"No turns parsed from {md_path}"

    history = []
    for i, turn in enumerate(turns):
        history.append({"role": "user", "content": turn.user_message})
        resp = client.post("/chat", json={"messages": history})

        assert resp.status_code == 200, (
            f"{trace_id} turn {i+1}: HTTP {resp.status_code} — {resp.text[:200]}"
        )

        body = resp.json()

        # Schema compliance
        assert isinstance(body.get("reply"), str), f"{trace_id} turn {i+1}: reply must be str"
        assert isinstance(body.get("recommendations"), list), (
            f"{trace_id} turn {i+1}: recommendations must be list"
        )
        assert isinstance(body.get("end_of_conversation"), bool), (
            f"{trace_id} turn {i+1}: end_of_conversation must be bool"
        )

        # No hallucinated URLs
        for rec in body["recommendations"]:
            assert rec.get("url") in valid_urls, (
                f"{trace_id} turn {i+1}: hallucinated URL {rec.get('url')}"
            )

        # Recall on turns with expected items
        if turn.expected_items:
            got = [r["name"] for r in body["recommendations"]]
            score = recall(got, turn.expected_items)
            assert score >= RECALL_THRESHOLD, (
                f"{trace_id} turn {i+1}: Recall@10 = {score:.2f} < {RECALL_THRESHOLD}\n"
                f"  Expected : {turn.expected_items}\n"
                f"  Got      : {got}"
            )

        # end_of_conversation on final turn
        if i == len(turns) - 1:
            assert body["end_of_conversation"] == turn.expected_end, (
                f"{trace_id} final turn: expected end_of_conversation="
                f"{turn.expected_end}, got {body['end_of_conversation']}"
            )

        history.append({"role": "assistant", "content": body["reply"]})
