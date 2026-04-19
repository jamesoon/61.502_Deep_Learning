"""Test more fixes for qwen3.5-9b empty output."""
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

# Fix 3: extra_body to disable thinking
print("=== Fix 3: extra_body enable_thinking=false ===")
try:
    r = c.chat.completions.create(
        model="qwen/qwen3.5-9b",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=2048,
        temperature=0,
        extra_body={"enable_thinking": False},
    )
    print("finish_reason:", r.choices[0].finish_reason)
    content = r.choices[0].message.content or ""
    print("content length:", len(content))
    print("content:", repr(content[:500]))
except Exception as e:
    print("ERROR:", e)

# Fix 4: chat_template_kwargs
print("\n=== Fix 4: chat_template_kwargs ===")
try:
    r = c.chat.completions.create(
        model="qwen/qwen3.5-9b",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=2048,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    print("finish_reason:", r.choices[0].finish_reason)
    content = r.choices[0].message.content or ""
    print("content length:", len(content))
    print("content:", repr(content[:500]))
except Exception as e:
    print("ERROR:", e)

# Fix 5: Very high max_tokens (32768) — brute force
print("\n=== Fix 5: max_tokens=32768 ===")
try:
    r = c.chat.completions.create(
        model="qwen/qwen3.5-9b",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=32768,
        temperature=0,
    )
    print("finish_reason:", r.choices[0].finish_reason)
    content = r.choices[0].message.content or ""
    print("content length:", len(content))
    print("content:", repr(content[:500]))
except Exception as e:
    print("ERROR:", e)
