"""
Two-stage hybrid pipeline with confidence gating:

  Stage 1: Cross-encoder classifier -> A/B/C/D + confidence
  Stage 2: LLM explainer (or ensemble voter if confidence is low)

Confidence gating logic:
  - HIGH confidence (>= threshold):  Use encoder answer, LLM explains it
  - LOW  confidence (<  threshold):  LLM reasons independently, ensemble vote

Usage:
  python pipeline.py \\
    --question "Which receptor?" \\
    --opa "M1" --opb "M2" --opc "M3" --opd "M4"

  python pipeline.py --skip-explain          # classifier only
  python pipeline.py --confidence-threshold 0.6  # adjust gating
  python pipeline.py --ensemble-mode always   # always let LLM vote
"""
from __future__ import annotations

import argparse
import json
import re

from infer_select import find_latest_ckpt, load_selector, predict_choice, resolve_device

LETTERS = ("A", "B", "C", "D")


def _extract_llm_answer(text: str) -> str | None:
    """Extract answer letter from LLM free-form response."""
    # Try structured format first: "Answer: B"
    m = re.search(r"Answer:\s*\**\s*([ABCD])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fall back to last standalone letter
    letters = re.findall(r"\b([ABCD])\b", text.upper())
    return letters[-1] if letters else None


def run_pipeline(
    question: str,
    options: list[str],
    *,
    exp: str | None = None,
    ckpt: str | None = None,
    models_dir: str = "./models",
    use_context: bool = True,
    device_select: str | None = None,
    skip_explain: bool = False,
    explainer_model: str = "Qwen/Qwen2.5-7B-Instruct",
    explainer_4bit: bool = False,
    device_explain: str | None = None,
    max_new_tokens: int = 512,
    confidence_threshold: float = 0.7,
    ensemble_mode: str = "auto",  # "auto", "always", "never"
) -> dict:
    """
    Run the hybrid pipeline.

    ensemble_mode:
      - "auto":   LLM votes only when encoder confidence < threshold
      - "always": LLM always votes independently (then we compare)
      - "never":  encoder answer only, LLM just explains
    """
    ckpt_path = ckpt or find_latest_ckpt(models_dir)
    model = load_selector(ckpt_path, device_select)

    sel = predict_choice(
        model,
        question,
        options,
        context=exp,
        use_context=use_context,
    )
    sel["checkpoint"] = ckpt_path
    sel["model_selector"] = "MCQAModel (cross-encoder + MLP)"

    out: dict = {
        "question": question,
        "options": {"A": options[0], "B": options[1], "C": options[2], "D": options[3]},
        "stage1_select": sel,
        "confidence_threshold": confidence_threshold,
        "ensemble_mode": ensemble_mode,
    }

    if skip_explain:
        out["stage2_explain"] = None
        out["final_answer"] = sel["predicted_letter"]
        out["answer_source"] = "encoder"
        return out

    from infer_explain import explain, build_user_prompt, load_llm, apply_chat_and_generate

    encoder_confident = sel["confidence"] >= confidence_threshold
    needs_ensemble = (ensemble_mode == "always" or
                      (ensemble_mode == "auto" and not encoder_confident))

    if needs_ensemble:
        # LLM reasons independently (no pre-selected answer)
        tokenizer, llm, dev = load_llm(
            explainer_model,
            device=device_explain or resolve_device(None),
            load_in_4bit=explainer_4bit,
        )

        # Independent reasoning prompt
        lines = [
            "You are a medical expert. Answer this multiple-choice question.",
            "Think step by step, then provide your answer.",
            "",
            f"Question: {question}",
            "",
            "Options:",
        ]
        for i, opt in enumerate(options):
            lines.append(f"  {LETTERS[i]}. {opt}")
        if exp and str(exp).strip():
            lines.append("")
            lines.append("Context:")
            lines.append(str(exp).strip())
        lines.append("")
        lines.append("Think step by step. Then respond in the format:")
        lines.append("Explanation: <your reasoning>")
        lines.append("Answer: <one of A, B, C, D>")

        independent_prompt = "\n".join(lines)
        llm_text = apply_chat_and_generate(
            tokenizer, llm, independent_prompt, max_new_tokens=max_new_tokens
        )
        llm_answer = _extract_llm_answer(llm_text)

        out["stage2_explain"] = {
            "explainer_model": explainer_model,
            "explanation": llm_text,
            "llm_independent_answer": llm_answer,
            "mode": "ensemble",
        }

        # Ensemble decision
        if llm_answer and llm_answer == sel["predicted_letter"]:
            # Both agree -> high confidence
            out["final_answer"] = sel["predicted_letter"]
            out["answer_source"] = "encoder+llm_agree"
        elif llm_answer and sel["confidence"] < 0.35:
            # Encoder very unsure -> trust LLM
            out["final_answer"] = llm_answer
            out["answer_source"] = "llm_override"
        else:
            # Disagreement but encoder has some confidence -> trust encoder
            out["final_answer"] = sel["predicted_letter"]
            out["answer_source"] = "encoder_preferred"

    else:
        # Encoder confident -> just explain its choice
        ex = explain(
            explainer_model,
            question,
            options,
            sel["predicted_letter"],
            evidence=exp,
            max_new_tokens=max_new_tokens,
            device=device_explain or resolve_device(None),
            load_in_4bit=explainer_4bit,
        )
        ex["mode"] = "explain_only"
        out["stage2_explain"] = ex
        out["final_answer"] = sel["predicted_letter"]
        out["answer_source"] = "encoder"

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid classifier + LLM pipeline")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--models-dir", type=str, default="./models")
    ap.add_argument("--device-select", type=str, default=None)
    ap.add_argument("--question", type=str, required=True)
    ap.add_argument("--opa", type=str, required=True)
    ap.add_argument("--opb", type=str, required=True)
    ap.add_argument("--opc", type=str, required=True)
    ap.add_argument("--opd", type=str, required=True)
    ap.add_argument("--exp", type=str, default="")
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--skip-explain", action="store_true")
    ap.add_argument("--explainer-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device-explain", type=str, default=None)
    ap.add_argument("--4bit", dest="fourbit", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--confidence-threshold", type=float, default=0.7,
                    help="Below this, trigger ensemble voting")
    ap.add_argument("--ensemble-mode", choices=["auto", "always", "never"],
                    default="auto",
                    help="auto=gate on confidence, always=LLM always votes, never=encoder only")
    args = ap.parse_args()

    opts = [args.opa, args.opb, args.opc, args.opd]
    use_ctx = not args.no_context
    exp = args.exp if args.exp.strip() else None

    result = run_pipeline(
        args.question,
        opts,
        exp=exp,
        ckpt=args.ckpt,
        models_dir=args.models_dir,
        use_context=use_ctx,
        device_select=args.device_select,
        skip_explain=args.skip_explain,
        explainer_model=args.explainer_model,
        explainer_4bit=args.fourbit,
        device_explain=args.device_explain,
        max_new_tokens=args.max_new_tokens,
        confidence_threshold=args.confidence_threshold,
        ensemble_mode=args.ensemble_mode,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
