"""Quick test to see what qwen3.5-9b actually outputs for an MCAT question."""
import json
from openai import OpenAI

c = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="x", timeout=120)

s = json.load(open("dataset_json/test_set_01/test_set_01_q0.json", "r", encoding="utf-8"))
ch = s.get("choices", {})
prompt = f"""Passage:
{s.get('passage', '').strip()}

Question:
{s.get('question', '').strip()}

Choices:
A. {ch.get('A', '')}
B. {ch.get('B', '')}
C. {ch.get('C', '')}
D. {ch.get('D', '')}

Return only one capital letter: A, B, C, or D.
"""

system = "You are taking an MCAT multiple-choice benchmark. Solve the question carefully, but return only one capital letter: A, B, C, or D. Do not explain your reasoning."

# Test with 2048 tokens
r = c.chat.completions.create(
    model="qwen/qwen3.5-9b",
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ],
    max_tokens=2048,
    temperature=0,
)

print("=== max_tokens=2048 ===")
print("finish_reason:", r.choices[0].finish_reason)
content = r.choices[0].message.content or ""
print("content length:", len(content))
print("FULL content repr:")
print(repr(content[:1000]))

# Test with higher tokens
r2 = c.chat.completions.create(
    model="qwen/qwen3.5-9b",
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ],
    max_tokens=4096,
    temperature=0,
)

print("\n=== max_tokens=4096 ===")
print("finish_reason:", r2.choices[0].finish_reason)
content2 = r2.choices[0].message.content or ""
print("content length:", len(content2))
print("FULL content repr:")
print(repr(content2[:2000]))
