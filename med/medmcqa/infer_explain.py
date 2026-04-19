"""
Stage 2: generate a short rationale with an instruction-tuned causal LM (e.g. Qwen2.5-Instruct).
Requires: pip install transformers accelerate
Optional (CUDA, tight VRAM): pip install bitsandbytes, use --4bit
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = ("A", "B", "C", "D")


def build_user_prompt(
    question: str,
    options: list[str],
    predicted_letter: str,
    evidence: str | None = None,
) -> str:
    lines = [
        "You are a medical educator. Explain briefly why the chosen answer is correct.",
        "Be factual and concise (under 200 words). Do not change the labeled answer.",
        "",
        f"Question: {question}",
        "",
        "Options:",
    ]
    for i, opt in enumerate(options):
        lines.append(f"  {LETTERS[i]}. {opt}")
    lines.append("")
    lines.append(f"Selected answer: {predicted_letter}")
    if evidence and str(evidence).strip():
        lines.append("")
        lines.append("Supporting context (may be incomplete):")
        lines.append(str(evidence).strip())
    lines.append("")
    lines.append("Provide a short explanation referencing the key medical concept.")
    return "\n".join(lines)


def load_llm(
    model_id: str,
    device: str | None = None,
    load_in_4bit: bool = False,
) -> tuple[Any, Any, str]:
    if device:
        dev = device
    elif torch.cuda.is_available():
        dev = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if load_in_4bit and dev == "cuda":
        try:
            from transformers import BitsAndBytesConfig

            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                quantization_config=qconfig,
                device_map="auto",
            )
        except ImportError as e:
            raise ImportError(
                "4-bit load requires: pip install bitsandbytes accelerate"
            ) from e
    else:
        dtype = torch.float16 if dev in ("cuda", "mps") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        model = model.to(dev)

    return tokenizer, model, dev


def apply_chat_and_generate(
    tokenizer,
    model,
    user_text: str,
    max_new_tokens: int = 512,
) -> str:
    messages = [{"role": "user", "content": user_text}]
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = user_text

    inputs = tokenizer(prompt, return_tensors="pt")
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
    inp_len = inputs["input_ids"].shape[-1]
    gen = out[0][inp_len:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def explain(
    model_id: str,
    question: str,
    options: list[str],
    predicted_letter: str,
    evidence: str | None = None,
    *,
    max_new_tokens: int = 512,
    device: str | None = None,
    load_in_4bit: bool = False,
) -> dict[str, Any]:
    tokenizer, model, _ = load_llm(model_id, device=device, load_in_4bit=load_in_4bit)
    prompt = build_user_prompt(question, options, predicted_letter, evidence)
    text = apply_chat_and_generate(
        tokenizer, model, prompt, max_new_tokens=max_new_tokens
    )
    return {
        "explainer_model": model_id,
        "explanation": text,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM explanation (stage 2)")
    ap.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF model id for instruct model",
    )
    ap.add_argument("--question", type=str, required=True)
    ap.add_argument("--opa", type=str, required=True)
    ap.add_argument("--opb", type=str, required=True)
    ap.add_argument("--opc", type=str, required=True)
    ap.add_argument("--opd", type=str, required=True)
    ap.add_argument("--letter", type=str, required=True, choices=list(LETTERS))
    ap.add_argument("--exp", type=str, default="", help="Optional evidence/context")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument(
        "--4bit",
        dest="fourbit",
        action="store_true",
        help="4-bit load on CUDA (requires bitsandbytes)",
    )
    args = ap.parse_args()

    opts = [args.opa, args.opb, args.opc, args.opd]
    out = explain(
        args.model,
        args.question,
        opts,
        args.letter,
        evidence=args.exp or None,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        load_in_4bit=args.fourbit,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
