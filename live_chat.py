import requests
import sys

URL = "https://assessment-recommender-ijgx.onrender.com/chat"
# If testing locally instead, use: URL = "http://127.0.0.1:8000/chat"

print("=====================================================")
print("🤖 SHL Assessment Recommender - Live Chat Terminal")
print("Type 'exit' or 'quit' to end the session.")
print("=====================================================\n")

# This list holds the running history since the API is stateless
messages = []

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Ending session.")
        sys.exit()

    # Append user message to history
    messages.append({"role": "user", "content": user_input})

    # Send the full history to the API
    try:
        response = requests.post(URL, json={"messages": messages})
        response.raise_for_status()
        data = response.json()
        
        reply = data.get("reply", "")
        recs = data.get("recommendations", [])
        ended = data.get("end_of_conversation", False)

        print(f"\nAgent: {reply}")
        
        # Display the shortlist if the agent provided one
        if recs:
            print("\n--- Shortlist ---")
            for i, rec in enumerate(recs, 1):
                print(f"  {i}. {rec['name']} [{rec['test_type']}]")
            print("-----------------")

        print("\n")

        # Append assistant message to history so the next turn remembers it
        messages.append({"role": "assistant", "content": reply})

        if ended:
            print("[Agent ended the conversation.]")
            messages = [] # Reset history for a new chat
            
    except requests.exceptions.RequestException as e:
        print(f"\n[Error connecting to API]: {e}\n")
        # Remove the failed message so we can try again
        messages.pop()