"""Test fixes for qwen3.5-9b empty output issue."""
import json
from openai import OpenAI

c = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="x", timeout=300)

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

# Fix 1: Much higher max_tokens
print("=== Fix 1: max_tokens=16384 ===")
r1 = c.chat.completions.create(
    model="qwen/qwen3.5-9b",
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ],
    max_tokens=16384,
    temperature=0,
)
print("finish_reason:", r1.choices[0].finish_reason)
content1 = r1.choices[0].message.content or ""
print("content length:", len(content1))
print("content:", repr(content1[:500]))

# Fix 2: Add /no_think instruction
print("\n=== Fix 2: /no_think in user message ===")
r2 = c.chat.completions.create(
    model="qwen/qwen3.5-9b",
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": prompt + "\n/no_think"},
    ],
    max_tokens=2048,
    temperature=0,
)
print("finish_reason:", r2.choices[0].finish_reason)
content2 = r2.choices[0].message.content or ""
print("content length:", len(content2))
print("content:", repr(content2[:500]))
