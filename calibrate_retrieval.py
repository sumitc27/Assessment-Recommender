"""
Retrieval calibration tool — run this ONCE to understand your score distributions.

What it does:
  1. Embeds a set of representative hiring queries (covering good matches, partial
     matches, and clearly out-of-scope requests)
  2. Runs each through FAISS and prints the top-5 scores + names
  3. Shows a score histogram so you can see where the natural break-points are
  4. Recommends a LOW_SIMILARITY_THRESHOLD value

  
Run:
    python calibrate_retrieval.py

Requires a running OpenAI API key in .env (same as the server).
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

import sys
sys.path.insert(0, str(Path(__file__).parent))

from app.data_loader import build_store, load_catalog, semantic_search_with_scores

# ---------------------------------------------------------------------------
# Probe queries — cover the spectrum from strong → weak → out-of-scope
# ---------------------------------------------------------------------------

PROBES: list[dict] = [
    # Strong matches — should score > 0.55
    {"label": "Java developer (strong)",      "query": "mid-level Java developer software engineer"},
    {"label": "Graduate analyst (strong)",    "query": "graduate financial analyst numerical reasoning entry level"},
    {"label": "Sales role (strong)",          "query": "sales manager personality behavior OPQ"},
    {"label": "Leadership exec (strong)",     "query": "executive director senior leadership selection benchmark"},
    {"label": "Contact centre (strong)",      "query": "entry level contact centre customer service agent"},

    # Partial matches — should score 0.35–0.55
    {"label": "HR business partner (partial)","query": "HR business partner stakeholder management mid professional"},
    {"label": "Nurse / healthcare (partial)", "query": "clinical nurse healthcare staff selection bilingual"},
    {"label": "Safety / plant (partial)",     "query": "manufacturing plant operator industrial safety dependability"},

    # Weak / out-of-scope — should score < 0.35
    {"label": "Rust developer (weak)",        "query": "Rust programming language systems engineer"},
    {"label": "Blockchain dev (weak)",        "query": "blockchain solidity Web3 smart contract developer"},
    {"label": "Chef / culinary (OOS)",        "query": "executive chef culinary arts kitchen management"},
    {"label": "Legal question (OOS)",         "query": "HIPAA compliance legal requirement testing staff"},
]


def embed(client: OpenAI, text: str) -> np.ndarray:
    response = client.embeddings.create(model="text-embedding-3-small", input=[text])
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def run_calibration():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("Loading catalog and building FAISS index …")
    catalog_path = next(
        p for p in [
            Path("shl_product_catalog.json"),
            Path("others/shl_product_catalog.json"),
        ]
        if p.exists()
    )
    catalog = load_catalog(catalog_path)
    store = build_store(catalog, client)
    print(f"  {len(catalog)} items indexed.\n")

    all_top_scores: list[float] = []
    bucket_counts: dict[str, int] = defaultdict(int)

    print("=" * 70)
    print(f"{'Label':<35} {'Top-1':>6}  {'Top-2':>6}  {'Top-3':>6}  Best match")
    print("=" * 70)

    for probe in PROBES:
        vec = embed(client, probe["query"])
        results = semantic_search_with_scores(store, vec, k=5)

        top_scores = [score for _, score in results[:3]]
        top1_name  = results[0][0].name if results else "—"
        top1_score = results[0][1] if results else 0.0

        all_top_scores.append(top1_score)

        s1 = f"{top_scores[0]:.3f}" if len(top_scores) > 0 else "—"
        s2 = f"{top_scores[1]:.3f}" if len(top_scores) > 1 else "—"
        s3 = f"{top_scores[2]:.3f}" if len(top_scores) > 2 else "—"
        label = probe["label"][:34]
        name  = top1_name[:28]
        print(f"{label:<35} {s1:>6}  {s2:>6}  {s3:>6}  {name}")

        # Bucket
        if top1_score >= 0.55:
            bucket_counts[">=0.55 (strong)"] += 1
        elif top1_score >= 0.40:
            bucket_counts["0.40-0.55 (good)"] += 1
        elif top1_score >= 0.30:
            bucket_counts["0.30-0.40 (weak)"] += 1
        else:
            bucket_counts["< 0.30 (gap)"] += 1

    print("=" * 70)
    print()

    # Histogram (ASCII-safe for Windows terminals)
    bucket_labels = [
        (">= 0.55 (strong)",    ">=0.55"),
        ("0.40-0.55 (good)",    "0.40-0.55"),
        ("0.30-0.40 (weak)",    "0.30-0.40"),
        ("< 0.30  (gap)",       "<0.30"),
    ]
    # Map bucket keys to display labels
    key_map = {
        ">= 0.55 (strong)":  ">=0.55 (strong)",
        "0.40-0.55 (good)":  "0.40-0.55 (good)",
        "0.30-0.40 (weak)":  "0.30-0.40 (weak)",
        "< 0.30  (gap)":     "< 0.30 (gap)",
    }
    print("Score distribution across probes:")
    for display, bucket_key in [
        (">= 0.55 (strong)", ">=0.55 (strong)"),
        ("0.40-0.55 (good)", "0.40-0.55 (good)"),
        ("0.30-0.40 (weak)", "0.30-0.40 (weak)"),
        ("< 0.30  (gap)",    "< 0.30 (gap)"),
    ]:
        count = bucket_counts.get(bucket_key, 0)
        bar   = "#" * count
        print(f"  {display:<25}  {bar}  ({count})")
    print()

    # Percentile summary
    arr = np.array(all_top_scores)
    print("Percentiles of top-1 scores:")
    for p in [10, 25, 50, 75, 90]:
        print(f"  p{p:02d}: {np.percentile(arr, p):.3f}")
    print(f"  min: {arr.min():.3f}   max: {arr.max():.3f}   mean: {arr.mean():.3f}")
    print()

    # Recommendation
    # We want the threshold to sit between "weak-but-in-catalog" and "truly out-of-scope"
    # A good heuristic: p25 of all top-1 scores, rounded down to nearest 0.05
    p25 = float(np.percentile(arr, 25))
    recommended = max(0.20, round(p25 / 0.05) * 0.05 - 0.05)
    current     = 0.28

    print("-" * 70)
    print(f"Current  LOW_SIMILARITY_THRESHOLD : {current}")
    print(f"Suggested value (p25 - 0.05)      : {recommended:.2f}")
    print()
    if recommended != current:
        print(f"  To apply: edit app/agents/raw_agent.py line 82:")
        print(f"    _LOW_SIMILARITY_THRESHOLD = {recommended:.2f}")
    else:
        print("  Current threshold looks reasonable — no change needed.")

    print()
    print("Interpretation guide:")
    print("  Raise the threshold -> agent reports gaps MORE often (stricter)")
    print("  Lower the threshold -> agent reports gaps LESS often (more permissive)")
    print("  Sweet spot: threshold should be ABOVE your worst in-catalog query score")
    print("              and BELOW clearly out-of-scope queries like 'chef' or 'HIPAA'.")


if __name__ == "__main__":
    run_calibration()
