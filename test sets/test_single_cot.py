"""Quick test: single CoT request to Qwen3-8B to measure timing."""
import time, json
from openai import OpenAI

c = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="x", timeout=120)

with open("dataset_json/test_set_01/test_set_01_q10.json") as f:
    s = json.load(f)

choices = s.get("choices", {})
prompt = (
    "Answer the MCAT Bio/Biochem question step by step.\n\n"
    "Rules:\n"
    "1. Identify the core biological or biochemical concept first.\n"
    "2. Give the final answer as a single option letter.\n\n"
    "Be concise.\n\n"
    f"Passage:\n{s.get('passage','').strip()[:500]}\n\n"
    f"Question:\n{s.get('question','').strip()}\n\n"
    f"Choices:\nA. {choices.get('A','')}\nB. {choices.get('B','')}\n"
    f"C. {choices.get('C','')}\nD. {choices.get('D','')}\n"
)

print(f"Sending request to qwen3-8b at {time.strftime('%H:%M:%S')}...")
t0 = time.time()
r = c.chat.completions.create(
    model="qwen3-8b",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=4096,
    temperature=0,
)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s | comp_tokens={r.usage.completion_tokens} | finish={r.choices[0].finish_reason}")
text = r.choices[0].message.content
print(f"Response length: {len(text)} chars")
print(f"Last 300 chars:\n...{text[-300:]}")
