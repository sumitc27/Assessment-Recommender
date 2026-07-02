import time
import requests

# The exact same test case payload (e.g., Test Case 1 sequence)
conversation_turns = [
    {"messages": [{"role": "user", "content": "I need a standard package for a retail sales representative."}]},
    {"messages": [
        {"role": "user", "content": "I need a standard package for a retail sales representative."},
        {"role": "assistant", "content": "What is the seniority level of the retail sales representative position?"},
        {"role": "user", "content": "entry level"}
    ]},
    {"messages": [
        {"role": "user", "content": "I need a standard package for a retail sales representative."},
        {"role": "assistant", "content": "What is the seniority level of the retail sales representative position?"},
        {"role": "user", "content": "entry level"},
        {"role": "assistant", "content": "What is the primary purpose of the assessment: selection, development, or screening?"},
        {"role": "user", "content": "selection"}
    ]}
]

def run_suite(engine_name):
    print(f"\n=== Testing Engine: {engine_name.upper()} ===")
    total_latency = 0
    
    for i, payload in enumerate(conversation_turns, 1):
        start_time = time.time()
        response = requests.post("http://127.0.0.1:8000/chat", json=payload)
        latency = time.time() - start_time
        total_latency += latency
        
        print(f"Turn {i} Latency: {latency:.2f}s | Status: {response.status_code}")
        
        # Verify schema compliance
        data = response.json()
        assert "reply" in data, "Missing reply field!"
        assert isinstance(data.get("recommendations"), list), "Recommendations must be a list!"
        assert isinstance(data.get("end_of_conversation"), bool), "end_of_conversation must be a boolean!"
        
    print(f"Average Turn Latency for {engine_name}: {total_latency / len(conversation_turns):.2f}s")

if __name__ == "__main__":
    import os
    # Detect which engine is active based on the running environment
    active_engine = os.getenv("RAG_ENGINE", "raw")
    run_suite(active_engine)