"""
Inference module — used by the webapp backend and for interactive testing.

Usage (CLI):
    python training/inference.py --model_id Qwen/Qwen3-30B-A3B --use_4bit
    python training/inference.py --adapter_path checkpoints/lora/final --use_4bit

Programmatic:
    from training.inference import MedQAInferenceEngine
    engine = MedQAInferenceEngine(model_id="Qwen/Qwen3-30B-A3B", use_4bit=True)
    result = engine.answer(question="...", options={"A": "...", "B": "...", "C": "...", "D": "..."})
"""

import os
import re
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MAX_GPU_MEMORY = "105GiB"
MAX_CPU_MEMORY = "200GiB"
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def detect_device() -> str:
    """Return best available device: 'cuda' > 'mps' > 'cpu'."""
    if torch.cuda.is_available():
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            "max_split_size_mb:512,expandable_segments:True",
        )
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

SYSTEM_PROMPT = (
    "You are a helpful tutor for pre-med students preparing for the MCAT. "
    "You answer multiple-choice questions with step-by-step reasoning."
)


def _format_prompt(question: str, options: dict) -> str:
    return (
        f"Question: {question}\n"
        f"Image Description: [not used in Stage 1]\n"
        f"Options:\n"
        f"A. {options['A']}\n"
        f"B. {options['B']}\n"
        f"C. {options['C']}\n"
        f"D. {options['D']}\n\n"
        f"Think step by step. Then respond in the format:\n"
        f"Explanation: ...\n"
        f"Answer: <one of A, B, C, D>"
    )


def _strip_thinking(text: str) -> str:
    """Remove Qwen3.5 <think>…</think> chain-of-thought blocks before parsing."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_answer(text: str) -> str | None:
    clean = _strip_thinking(text)
    m = re.search(r"Answer:\s*\**\s*([ABCD])\b", clean, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    letters = re.findall(r"\b([ABCD])\b", clean.upper())
    return letters[-1] if letters else None


def _extract_explanation(text: str) -> str:
    clean = _strip_thinking(text)
    m = re.search(r"Explanation:\s*(.+?)(?:\nAnswer:|$)", clean, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else clean.strip()


class MedQAInferenceEngine:
    """
    Wrapper for inference with a base or fine-tuned Qwen model.

    Args:
        model_id:     HuggingFace model ID (default Qwen/Qwen3-30B-A3B)
        adapter_path: Path to LoRA adapter dir or full SFT checkpoint dir.
                      Pass None to use the base model zero-shot.
        use_4bit:     Load in 4-bit NF4 (recommended for QLoRA adapters; saves VRAM)
    """

    def __init__(
        self,
        model_id: str = "google/gemma-3-4b-it",
        adapter_path: str | None = None,
        use_4bit: bool = False,
    ):
        device = detect_device()

        # 4-bit quantization requires bitsandbytes which only supports CUDA
        if use_4bit and device != "cuda":
            print(f"[Inference] Warning: 4-bit quantization requires CUDA; ignoring use_4bit on {device}")
            use_4bit = False

        bnb_config = None
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # dtype: bfloat16 on CUDA, float16 on MPS (bfloat16 unsupported), float32 on CPU
        if device == "cuda":
            torch_dtype = torch.bfloat16 if not use_4bit else None
        elif device == "mps":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        attn_impl = "eager"  # safe default; sdpa causes issues on MPS and Qwen3.5 hybrid arch

        # On CUDA with large models, enforce memory cap; on MPS/CPU, use device_map directly
        extra_kwargs = {}
        if device == "cuda":
            extra_kwargs["max_memory"] = {0: MAX_GPU_MEMORY, "cpu": MAX_CPU_MEMORY}
            extra_kwargs["device_map"] = "auto"
        else:
            extra_kwargs["device_map"] = {"": device}

        tok_src = adapter_path if adapter_path else model_id
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            **extra_kwargs,
        )

        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()
        print(f"[Inference] Model ready on {next(self.model.parameters()).device}")

    def answer(
        self,
        question: str,
        options: dict,          # {"A": "...", "B": "...", "C": "...", "D": "..."}
        max_new_tokens: int = 512,
    ) -> dict:
        """
        Returns:
            {
                "answer": "B",                      # predicted letter
                "explanation": "Because ...",       # extracted explanation
                "raw_output": "<full model text>",
            }
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _format_prompt(question, options)},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=1536
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,   # Qwen3.5 recommended for precise/coding tasks (thinking mode)
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return {
            "answer": _extract_answer(generated),
            "explanation": _extract_explanation(generated),
            "raw_output": generated,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen3.5-35B-A3B")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--use_4bit", action="store_true", default=True)
    args = parser.parse_args()

    engine = MedQAInferenceEngine(args.model_id, args.adapter_path, args.use_4bit)

    # Example question
    result = engine.answer(
        question="Which hormone is most likely responsible for the observed increase in blood glucose after a high-stress event?",
        options={
            "A": "Insulin",
            "B": "Cortisol",
            "C": "Glucagon",
            "D": "Epinephrine",
        },
    )
    print(f"\nAnswer:      {result['answer']}")
    print(f"Explanation: {result['explanation']}")
