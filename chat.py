"""
Interactive CLI to chat with the running SHL recommender agent.

Usage (server must be running in another terminal):
    python chat.py
"""

import httpx

BASE_URL = "http://localhost:8000"
history = []

print("SHL Assessment Recommender — interactive test")
print("Type 'quit' to exit, 'reset' to start a new conversation.")
print("-" * 55)

while True:
    user_input = input("\nYou: ").strip()

    if not user_input:
        continue
    if user_input.lower() == "quit":
        break
    if user_input.lower() == "reset":
        history = []
        print("— conversation reset —")
        continue

    history.append({"role": "user", "content": user_input})

    try:
        resp = httpx.post(
            f"{BASE_URL}/chat",
            json={"messages": history},
            timeout=35,
        )
        resp.raise_for_status()
        body = resp.json()
    except httpx.TimeoutException:
        print("Agent: [timeout — took longer than 35s]")
        history.pop()
        continue
    except Exception as exc:
        print(f"Agent: [error — {exc}]")
        history.pop()
        continue

    reply = body["reply"]
    recs = body["recommendations"]
    eoc = body["end_of_conversation"]

    print(f"\nAgent: {reply}")

    if recs:
        print("\n  Recommendations:")
        for i, r in enumerate(recs, 1):
            print(f"    {i}. {r['name']} [{r['test_type']}]")
            print(f"       {r['url']}")

    if eoc:
        print("\n— conversation complete —")
        history = []
    else:
        history.append({"role": "assistant", "content": reply})
