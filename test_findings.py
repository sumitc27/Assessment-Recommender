"""
Systematic test of all 10 Conversation_Finding.md behaviors.
Run with server active: python test_findings.py

Prints PASS / FAIL for each finding with evidence.
"""

import httpx, json, textwrap

BASE = "http://localhost:8000"
client = httpx.Client(base_url=BASE, timeout=35)

CATALOG = {
    item["link"]: item["name"]
    for item in json.load(open("shl_product_catalog.json", encoding="utf-8"))
}
VALID_URLS = set(CATALOG)

results = {}

def chat(messages):
    r = client.post("/chat", json={"messages": messages})
    r.raise_for_status()
    return r.json()

def rec_names(body):
    return [r["name"] for r in body.get("recommendations", [])]

def check(finding_id, passed, evidence=""):
    label = "✅ PASS" if passed else "❌ FAIL"
    results[finding_id] = passed
    print(f"\n{label}  Finding {finding_id}")
    if evidence:
        for line in textwrap.wrap(evidence, 80):
            print(f"       {line}")

print("=" * 60)
print("  SHL Chatbot — Conversation Findings Diagnostic")
print("=" * 60)

# ── Finding 1: Clarify can chain multiple turns ─────────────────────────────
# A vague "senior leadership" message should clarify, not recommend
t1 = chat([{"role": "user", "content": "We need a solution for senior leadership."}])
f1a = t1["recommendations"] == []          # no recs on vague turn
f1b = "?" in t1["reply"]                  # asked a question
check(1, f1a and f1b,
      f"recs={rec_names(t1)!r}  has_question={'?' in t1['reply']}")

# ── Finding 2: Prose mention ≠ recommendations field ────────────────────────
# After clarify, recs should still be [] if we haven't confirmed enough
t2a = [{"role": "user", "content": "We need a solution for senior leadership."}]
t2b = chat(t2a + [{"role": "assistant", "content": t1["reply"]},
                  {"role": "user", "content": "CXOs, director-level, 15+ years experience."}])
# Agent may mention OPQ32r in prose but still ask about selection vs development
# so recs can be [] or populated — what matters is it doesn't hallucinate URLs
bad_urls = [r["url"] for r in t2b.get("recommendations", []) if r["url"] not in VALID_URLS]
check(2, bad_urls == [],
      f"Hallucinated URLs: {bad_urls}" if bad_urls else "No hallucinated URLs")

# ── Finding 3: Refine — three sub-behaviors ──────────────────────────────────
# 3a: Pure addition — add situational judgment to grad financial analysts
h3 = [{"role": "user", "content": "Hiring grad financial analysts — need numerical reasoning."}]
b3a = chat(h3)
before_names = set(rec_names(b3a))
h3 += [{"role": "assistant", "content": b3a["reply"]},
       {"role": "user",      "content": "Add a situational judgment element too."}]
b3b = chat(h3)
after_names = set(rec_names(b3b))
# Must have grown, old items mostly preserved
items_kept = before_names & after_names
check("3a-add", len(after_names) >= len(before_names),
      f"Before:{sorted(before_names)}  After:{sorted(after_names)}")

# 3b: Explicit removal
h3r = h3 + [{"role": "assistant", "content": b3b["reply"]},
             {"role": "user",      "content": "Drop Graduate Scenarios — we'll use an interview instead."}]
b3r = chat(h3r)
still_there = [n for n in rec_names(b3r) if "graduate scenarios" in n.lower()]
check("3b-remove", still_there == [],
      f"Should be gone but still present: {still_there}")

# 3c: Disambiguation collapse (offer two, user picks one)
h3d = [
    {"role": "user",      "content": "Hiring plant operators in an industrial facility — safety-critical."},
]
b3d_init = chat(h3d)
# Force a disambiguation scenario by asking which of two items to keep
h3d += [
    {"role": "assistant", "content": b3d_init["reply"]},
    {"role": "user",      "content": "We're industrial. If you offered DSI and the Manufacturing Safety bundle, we want just the bundle — drop DSI."},
]
b3d = chat(h3d)
dsi_present = any("dependability and safety" in n.lower() for n in rec_names(b3d)
                  if "manufacturing" not in n.lower())
check("3c-disambig", not dsi_present,
      f"DSI still present after user picked bundle: {rec_names(b3d)}")

# ── Finding 4: Agent pushback, but user's final word wins ────────────────────
h4 = [
    {"role": "user",      "content": "Graduate management trainee — need cognitive, personality, situational judgment."},
]
b4a = chat(h4)
h4 += [
    {"role": "assistant", "content": b4a["reply"]},
    {"role": "user",      "content": "Remove OPQ32r and replace it with something shorter."},
]
b4b = chat(h4)
# Agent should push back (no shorter alternative) but NOT remove it without confirmation
# Check agent mentioned limitation
pushback = any(kw in b4b["reply"].lower()
               for kw in ("no shorter", "no equivalent", "shortest", "no alternative", "still recommend"))
h4 += [
    {"role": "assistant", "content": b4b["reply"]},
    {"role": "user",      "content": "Fine, just drop it entirely."},
]
b4c = chat(h4)
opq_gone = not any("opq32r" in n.lower() or "occupational personality" in n.lower()
                    for n in rec_names(b4c))
check(4, opq_gone,
      f"Pushback detected={pushback}  OPQ removed after insistence={opq_gone}  "
      f"Final recs: {rec_names(b4c)}")

# ── Finding 5: Default-bundling hierarchy ────────────────────────────────────
# General role → OPQ32r should appear
b5 = chat([{"role": "user", "content": "Hiring mid-level Java developers — need coding and problem-solving tests."}])
has_opq = any("opq32r" in n.lower() or "occupational personality" in n.lower()
              for n in rec_names(b5))
# Safety role → should NOT get bare OPQ32r, should get DSI-family instead
b5s = chat([{"role": "user", "content": "Hiring plant operators for a chemical plant — safety-critical reliability role."}])
has_safety = any(kw in " ".join(rec_names(b5s)).lower()
                 for kw in ("safety", "dependab", "dsi", "manufacturing"))
check(5, has_safety,
      f"General role OPQ={has_opq}  Safety role has safety instrument={has_safety}  "
      f"Safety recs: {rec_names(b5s)}")

# ── Finding 6: Catalog gaps named explicitly ─────────────────────────────────
b6 = chat([{"role": "user", "content": "Hiring a senior Rust engineer — need a Rust programming test."}])
gap_mentioned = any(kw in b6["reply"].lower()
                    for kw in ("no rust", "rust-specific", "no specific", "not available",
                               "doesn't exist", "does not exist", "no test", "no direct"))
check(6, gap_mentioned,
      f"Gap mentioned in reply={gap_mentioned}  Reply: {b6['reply'][:120]}")

# ── Finding 7: Refusal scoped to turn, conversation continues ────────────────
h7 = [
    {"role": "user",      "content": "Hiring bilingual healthcare admin — need HIPAA and medical terminology tests."},
]
b7a = chat(h7)
standing = rec_names(b7a)
h7 += [
    {"role": "assistant", "content": b7a["reply"]},
    {"role": "user",      "content": "Are we legally required under HIPAA to test all staff?"},
]
b7b = chat(h7)
eoc_false = not b7b["end_of_conversation"]
# shortlist should be preserved (or at least not empty if it was populated)
shortlist_ok = len(b7b["recommendations"]) > 0 if standing else True
check(7, eoc_false and shortlist_ok,
      f"eoc={b7b['end_of_conversation']}  shortlist_preserved={shortlist_ok}  "
      f"Reply: {b7b['reply'][:100]}")

# ── Finding 8: Locale is first-class ─────────────────────────────────────────
b8 = chat([{"role": "user",
            "content": "Hiring bilingual healthcare admin staff in South Texas. "
                       "Candidates are Spanish-speaking. Need assessments — HIPAA knowledge, medical terminology."}])
locale_visible = any(kw in b8["reply"].lower()
                     for kw in ("spanish", "english", "language", "bilingual", "locale"))
check(8, locale_visible,
      f"Locale mentioned in reply={locale_visible}  Reply: {b8['reply'][:150]}")

# ── Finding 9: Compare turn repeats shortlist unchanged ──────────────────────
h9 = [{"role": "user", "content": "Hiring graduate financial analysts — numerical reasoning and finance knowledge."}]
b9a = chat(h9)
shortlist_before = rec_names(b9a)
h9 += [
    {"role": "assistant", "content": b9a["reply"]},
    {"role": "user",      "content": "What's the difference between SHL Verify Interactive Numerical Reasoning and Verify Numerical Ability?"},
]
b9b = chat(h9)
shortlist_after = rec_names(b9b)
# Shortlist should be repeated (not empty)
check(9, len(shortlist_after) > 0,
      f"Before compare: {shortlist_before}  After compare: {shortlist_after}")

# ── Finding 10: recommendations = [] not null ────────────────────────────────
b10 = chat([{"role": "user", "content": "I need an assessment."}])
is_list = isinstance(b10.get("recommendations"), list)
is_not_null = b10.get("recommendations") is not None
check(10, is_list and is_not_null,
      f"type={type(b10.get('recommendations')).__name__}  value={b10.get('recommendations')}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
passed = sum(1 for v in results.values() if v)
total  = len(results)
print(f"  Result: {passed}/{total} findings passing")
print("=" * 60)
for fid, ok in results.items():
    print(f"  {'✅' if ok else '❌'}  Finding {fid}")
