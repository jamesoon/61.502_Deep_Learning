"""Test what token/performance data LM Studio API returns."""
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

# Test with gemma (fast) to not interfere with running eval
r = c.chat.completions.create(
    model="google-gemma-3-4b-it-qat-small",
    messages=[
        {"role": "system", "content": "You are taking an MCAT benchmark. Return only A, B, C, or D."},
        {"role": "user", "content": prompt},
    ],
    max_tokens=2048,
    temperature=0,
)

print("=== Response object fields ===")
print("usage:", r.usage)
if r.usage:
    print("  prompt_tokens:", r.usage.prompt_tokens)
    print("  completion_tokens:", r.usage.completion_tokens)
    print("  total_tokens:", r.usage.total_tokens)
    # Check for extra fields
    for attr in dir(r.usage):
        if not attr.startswith('_'):
            print(f"  usage.{attr}:", getattr(r.usage, attr, None))

print("\nmodel:", r.model)
print("id:", r.id)
print("content:", repr(r.choices[0].message.content))
print("finish_reason:", r.choices[0].finish_reason)

# Check raw response for any extra fields
print("\n=== Full response dict ===")
print(r.model_dump_json(indent=2)[:2000])
