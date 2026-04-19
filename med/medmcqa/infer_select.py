"""
Stage 1: load a cross-encoder MCQ checkpoint and predict A/B/C/D.

Works with both legacy per-option checkpoints and new cross-encoder checkpoints.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import torch

from model import MCQAModel

LETTERS = ("A", "B", "C", "D")


def _arg_get(model: MCQAModel, key: str, default=None):
    a = model.args
    if isinstance(a, dict):
        return a.get(key, default)
    return getattr(a, key, default)


def resolve_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_latest_ckpt(models_root: str | Path, pattern: str = "**/*.pt") -> str:
    root = Path(models_root)
    # Try .pt first (new format), then .ckpt (legacy)
    for ext_pattern in (pattern, "**/*.ckpt"):
        paths = sorted(glob.glob(str(root / ext_pattern), recursive=True))
        if paths:
            return paths[-1]
    raise FileNotFoundError(f"No checkpoints under {root}")


def load_selector(ckpt_path: str, device: str | None = None) -> MCQAModel:
    device = resolve_device(device)
    model = MCQAModel.load_checkpoint(ckpt_path, device=device)
    model.args["device"] = device
    model = model.to(device)
    model.eval()
    return model


def predict_choice(
    model: MCQAModel,
    question: str,
    options: list[str],
    *,
    context: str | None = None,
    use_context: bool = True,
) -> dict[str, Any]:
    """Return predicted index 0-3, letter, softmax probs, and confidence."""
    if len(options) != 4:
        raise ValueError("Expected exactly 4 options (opa..opd order).")

    if use_context and context is not None and str(context).strip():
        batch = [(str(context), question, options, 0)]
    else:
        batch = [(question, options, 0)]

    max_len = _arg_get(model, "max_len", 512)
    inputs, _ = model.process_batch(batch, model.tokenizer, max_len=max_len)
    dev = _arg_get(model, "device", "cpu")
    for k in list(inputs.keys()):
        inputs[k] = inputs[k].to(dev)

    with torch.no_grad():
        logits = model(**inputs)
    probs = torch.softmax(logits, dim=-1)[0]
    pred_idx = int(torch.argmax(probs).item())
    return {
        "predicted_index": pred_idx,
        "predicted_letter": LETTERS[pred_idx],
        "confidence": float(probs[pred_idx].item()),
        "probs": {LETTERS[i]: float(probs[i].item()) for i in range(4)},
    }


def predict_batch(
    model: MCQAModel,
    questions: list[str],
    options_list: list[list[str]],
    *,
    contexts: list[str | None] | None = None,
    use_context: bool = True,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Batch prediction for evaluation."""
    if contexts is None:
        contexts = [None] * len(questions)

    results = []
    max_len = _arg_get(model, "max_len", 512)
    dev = _arg_get(model, "device", "cpu")

    for start in range(0, len(questions), batch_size):
        end = min(start + batch_size, len(questions))
        batch = []
        for i in range(start, end):
            ctx = contexts[i]
            if use_context and ctx is not None and str(ctx).strip():
                batch.append((str(ctx), questions[i], options_list[i], 0))
            else:
                batch.append((questions[i], options_list[i], 0))

        inputs, _ = model.process_batch(batch, model.tokenizer, max_len=max_len)
        for k in list(inputs.keys()):
            inputs[k] = inputs[k].to(dev)

        with torch.no_grad():
            logits = model(**inputs)
        probs = torch.softmax(logits, dim=-1)

        for j in range(probs.size(0)):
            p = probs[j]
            pred_idx = int(torch.argmax(p).item())
            results.append({
                "predicted_index": pred_idx,
                "predicted_letter": LETTERS[pred_idx],
                "confidence": float(p[pred_idx].item()),
                "probs": {LETTERS[k]: float(p[k].item()) for k in range(4)},
            })

    return results


def main() -> None:
    p = argparse.ArgumentParser(description="MCQ cross-encoder inference (stage 1)")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--models-dir", type=str, default="./models")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--use-context", action="store_true", default=True)
    p.add_argument("--no-context", action="store_true")
    p.add_argument("--question", type=str, required=True)
    p.add_argument("--opa", type=str, required=True)
    p.add_argument("--opb", type=str, required=True)
    p.add_argument("--opc", type=str, required=True)
    p.add_argument("--opd", type=str, required=True)
    p.add_argument("--exp", type=str, default="")
    args = p.parse_args()

    use_context = args.use_context and not args.no_context
    ckpt = args.ckpt or find_latest_ckpt(args.models_dir)
    model = load_selector(ckpt, args.device)
    out = predict_choice(
        model,
        args.question,
        [args.opa, args.opb, args.opc, args.opd],
        context=args.exp or None,
        use_context=use_context,
    )
    out["checkpoint"] = ckpt
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
