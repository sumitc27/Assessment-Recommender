"""
Evaluation harness — replays all 10 labeled traces against the live server
and reports schema compliance, Recall@10, and final-turn end_of_conversation.

Run:
    python evaluate.py                  # assumes server on localhost:8000
    python evaluate.py --url http://...  # custom server URL
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import httpx

TRACES_DIR = Path("GenAI_SampleConversations")
CATALOG_PATH = Path("shl_product_catalog.json")


# ---------------------------------------------------------------------------
# Trace parsing
# ---------------------------------------------------------------------------

@dataclass
class TraceTurn:
    user_message: str
    expected_items: list[str]        # product names from the table (empty = no rec this turn)
    expected_end: bool


def parse_trace(md_path: Path) -> list[TraceTurn]:
    """
    Parse one C*.md file into a list of TraceTurn objects.

    The markdown format used in the traces is:
      **User**
      > <message>
      | # | Name | ... |   ← optional table of recommendations
      `end_of_conversation`: **true/false**
    """
    text = md_path.read_text(encoding="utf-8")
    # Split on turn headings: ### Turn N
    turn_blocks = re.split(r"###\s+Turn\s+\d+", text)[1:]  # skip preamble

    turns: list[TraceTurn] = []
    for block in turn_blocks:
        # User message
        user_match = re.search(r"\*\*User\*\*\s*\n+>\s*(.+)", block)
        if not user_match:
            continue
        user_msg = user_match.group(1).strip()

        # Recommendations table — grab "Name" column (2nd column, index 1)
        # Table rows look like: | 1 | Product Name | ... |
        table_rows = re.findall(r"^\|\s*\d+\s*\|\s*([^|]+)\|", block, re.MULTILINE)
        expected_items = [r.strip() for r in table_rows if r.strip()]

        # end_of_conversation flag
        end_match = re.search(
            r"`end_of_conversation`.*?\*\*(true|false)\*\*", block, re.IGNORECASE
        )
        expected_end = end_match.group(1).lower() == "true" if end_match else False

        turns.append(TraceTurn(
            user_message=user_msg,
            expected_items=expected_items,
            expected_end=expected_end,
        ))
    return turns


# ---------------------------------------------------------------------------
# Recall@10
# ---------------------------------------------------------------------------

def recall_at_10(got_names: list[str], expected_names: list[str]) -> float:
    if not expected_names:
        return 1.0
    got_lower = {n.lower() for n in got_names}
    hits = sum(1 for e in expected_names if e.lower() in got_lower)
    return hits / len(expected_names)


# ---------------------------------------------------------------------------
# Single trace replay
# ---------------------------------------------------------------------------

def replay_trace(
    trace_id: str,
    turns: list[TraceTurn],
    base_url: str,
) -> dict:
    """
    Replay one trace turn-by-turn against the server.
    Returns a result dict with pass/fail counts and per-turn details.
    """
    history = []
    result = {
        "trace": trace_id,
        "turns": len(turns),
        "schema_failures": 0,
        "recall_scores": [],
        "end_of_conv_correct": True,
        "details": [],
    }

    valid_urls: set[str] = set()

    for i, turn in enumerate(turns):
        history.append({"role": "user", "content": turn.user_message})

        try:
            resp = httpx.post(
                f"{base_url}/chat",
                json={"messages": history},
                timeout=30,
            )
            resp.raise_for_status()
            body = resp.json()
        except Exception as exc:
            result["schema_failures"] += 1
            result["details"].append({"turn": i + 1, "error": str(exc)})
            history.append({"role": "assistant", "content": "(error)"})
            continue

        # Schema checks
        schema_ok = (
            isinstance(body.get("reply"), str)
            and isinstance(body.get("recommendations"), list)
            and isinstance(body.get("end_of_conversation"), bool)
        )
        if not schema_ok:
            result["schema_failures"] += 1

        # URL hallucination check (lazy-load valid URLs once)
        if not valid_urls:
            catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
            valid_urls = {item["link"] for item in catalog}

        bad_urls = [
            r["url"] for r in body.get("recommendations", [])
            if r.get("url") not in valid_urls
        ]

        # Recall on the final turn (or any turn that has expected items)
        got_names = [r["name"] for r in body.get("recommendations", [])]
        if turn.expected_items:
            score = recall_at_10(got_names, turn.expected_items)
            result["recall_scores"].append(score)

        # end_of_conversation correctness on last turn
        if i == len(turns) - 1:
            result["end_of_conv_correct"] = (
                body.get("end_of_conversation") == turn.expected_end
            )

        turn_detail = {
            "turn": i + 1,
            "schema_ok": schema_ok,
            "bad_urls": bad_urls,
            "got_items": got_names,
            "expected_items": turn.expected_items,
        }
        if turn.expected_items:
            turn_detail["recall"] = recall_at_10(got_names, turn.expected_items)
        result["details"].append(turn_detail)

        history.append({"role": "assistant", "content": body.get("reply", "")})

    avg_recall = (
        sum(result["recall_scores"]) / len(result["recall_scores"])
        if result["recall_scores"] else None
    )
    result["avg_recall"] = avg_recall
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate SHL recommender against traces")
    parser.add_argument("--url", default="http://localhost:8000", help="Server base URL")
    args = parser.parse_args()

    # Health check first
    try:
        health = httpx.get(f"{args.url}/health", timeout=10)
        assert health.json() == {"status": "ok"}, f"Unexpected health: {health.text}"
        print(f"✓ Server healthy at {args.url}")
    except Exception as exc:
        print(f"✗ Server not reachable: {exc}")
        sys.exit(1)

    trace_files = sorted(TRACES_DIR.glob("C*.md"))
    if not trace_files:
        print(f"✗ No trace files found in {TRACES_DIR}/")
        sys.exit(1)

    all_results = []
    for tf in trace_files:
        turns = parse_trace(tf)
        result = replay_trace(tf.stem, turns, args.url)
        all_results.append(result)

        recall_str = (
            f"{result['avg_recall']:.2f}" if result["avg_recall"] is not None else "n/a"
        )
        status = "✓" if result["schema_failures"] == 0 else "✗"
        end_str = "✓" if result["end_of_conv_correct"] else "✗ end_of_conv wrong"
        print(
            f"  {status} {tf.stem}  "
            f"recall={recall_str}  "
            f"schema_failures={result['schema_failures']}  "
            f"eoc={end_str}"
        )

    # Summary
    total_failures = sum(r["schema_failures"] for r in all_results)
    recall_scores = [r["avg_recall"] for r in all_results if r["avg_recall"] is not None]
    mean_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

    print()
    print("─" * 50)
    print(f"Traces evaluated : {len(all_results)}")
    print(f"Schema failures  : {total_failures}")
    print(f"Mean Recall@10   : {mean_recall:.2f}")
    eoc_correct = sum(1 for r in all_results if r["end_of_conv_correct"])
    print(f"EOC correct      : {eoc_correct}/{len(all_results)}")

    if total_failures > 0 or mean_recall < 0.6:
        sys.exit(1)


if __name__ == "__main__":
    main()
