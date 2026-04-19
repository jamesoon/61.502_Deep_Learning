"""
Grade Lambda — Step Functions task
====================================
Called once per question in the Map state.

Input (from Step Functions map item):
{
  "submissionId": str,
  "questionId": str,
  "question": str,
  "option_a" .. "option_d": str,
  "correct_answer": "A"|"B"|"C"|"D",
  "student_answer": "A"|"B"|"C"|"D",
  "is_correct": bool
}

Output:
{ ...same fields... , "explanation": str }

HF Inference strategy:
- Primary: HF Inference API → jamezoon/gemma-3-4b-it-medmcqa-lora (LLM explanation)
- The model generates an explanation for why the correct answer is right.
- When DeBERTa cross-encoder is published, swap HF_GRADE_MODEL env var to use it
  for answer selection and only call Gemma for explanation.

DEMO_MODE
---------
When the SSM parameter `/medmcqa/demo_mode` is "1" (or env DEMO_MODE=1),
the HF call is bypassed and pre-baked explanations are returned from the
bundled `demo_explanations.json` file. This keeps the live demo deterministic,
sub-second, and zero-cost while still exercising the Step Functions Map
fan-out, IAM, DynamoDB write path, and Cognito auth.

Override the SSM parameter live with:
  aws ssm put-parameter --name /medmcqa/demo_mode --type String \
      --value 1 --overwrite --region ap-southeast-1

Set to "0" (or delete the parameter) to fall back to the real HF endpoint.
"""

import hashlib
import json
import os
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

import boto3

HF_EXPLAIN_MODEL = os.environ.get("HF_EXPLAIN_MODEL", "jamezoon/gemma-3-4b-it-medmcqa-lora")
HF_GRADE_MODEL   = os.environ.get("HF_GRADE_MODEL", "")   # set once DeBERTa is published
REGION           = os.environ.get("REGION", "ap-southeast-1")
DEFAULT_EVAL_MODELS = (
    "jamezoon/gemma-3-4b-it-medmcqa-lora,jamezoon/deberta-v3-large-medmcqa"
)

# DEMO_MODE: env var is the immediate fallback; SSM is the authoritative source.
# Cached at module-init so that the SSM lookup happens once per warm container.
_DEMO_MODE_ENV = os.environ.get("DEMO_MODE", "0") == "1"

ssm = boto3.client("ssm", region_name=REGION)
_hf_token_cache = None
_demo_mode_cache: bool | None = None
_demo_explanations: dict | None = None
_eval_models_cache: list | None = None


def _get_eval_models() -> list:
    """CSV-formatted HF model IDs from /medmcqa/eval_models; cached per container."""
    global _eval_models_cache
    if _eval_models_cache is not None:
        return _eval_models_cache
    raw = ""
    try:
        resp = ssm.get_parameter(Name="/medmcqa/eval_models")
        raw = resp["Parameter"]["Value"]
    except Exception:
        raw = os.environ.get("EVAL_MODELS", DEFAULT_EVAL_MODELS)
    _eval_models_cache = [m.strip() for m in raw.split(",") if m.strip()]
    return _eval_models_cache


def _get_hf_token() -> str:
    global _hf_token_cache
    if _hf_token_cache:
        return _hf_token_cache
    try:
        resp = ssm.get_parameter(Name="/medmcqa/hf_token", WithDecryption=True)
        _hf_token_cache = resp["Parameter"]["Value"]
    except Exception:
        _hf_token_cache = os.environ.get("HF_TOKEN", "")
    return _hf_token_cache


def _is_demo_mode() -> bool:
    """SSM-backed demo flag with env-var fallback. Cached per warm container."""
    global _demo_mode_cache
    if _demo_mode_cache is not None:
        return _demo_mode_cache
    try:
        resp = ssm.get_parameter(Name="/medmcqa/demo_mode")
        _demo_mode_cache = resp["Parameter"]["Value"].strip() == "1"
    except Exception:
        # ParameterNotFound or any other SSM error → fall through to env var
        _demo_mode_cache = _DEMO_MODE_ENV
    return _demo_mode_cache


def _load_demo_explanations() -> dict:
    """Load the canned explanation map bundled with the Lambda."""
    global _demo_explanations
    if _demo_explanations is not None:
        return _demo_explanations
    bundle = Path(__file__).parent / "demo_explanations.json"
    try:
        with bundle.open() as f:
            _demo_explanations = json.load(f)
    except Exception:
        _demo_explanations = {}
    return _demo_explanations


def _demo_explanation(event: dict) -> str:
    """Return a canned explanation for demo mode. Falls back to a templated
    answer if the questionId is not in the bundle."""
    bundle = _load_demo_explanations()
    qid    = event.get("questionId", "")
    keyed  = bundle.get(qid)
    if keyed:
        return keyed
    # Fallback templated explanation — keeps the demo deterministic even
    # for ad-hoc questions admins create live.
    correct = event.get("correct_answer", "?")
    is_correct = event.get("is_correct", False)
    if is_correct:
        return (
            f"[demo] The correct answer is {correct}. This option best matches the "
            f"clinical/anatomical concept being tested. (Pre-baked explanation; toggle "
            f"DEMO_MODE off to call the live Gemma-3-4B-it MedMCQA LoRA adapter.)"
        )
    student = event.get("student_answer", "?")
    return (
        f"[demo] The correct answer is {correct}. {student} is incorrect because it "
        f"does not satisfy the key diagnostic criterion in the question stem. "
        f"(Pre-baked explanation for demo mode.)"
    )


def _hf_post(model_id: str, payload: dict, timeout: int = 45) -> dict:
    """POST to HF Inference API. Raises on HTTP error."""
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    token = _get_hf_token()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _build_explanation_prompt(question: str, options: dict,
                               correct_letter: str, is_correct: bool,
                               student_letter: str) -> str:
    opts = "\n".join(f"{k.upper()}. {v}" for k, v in options.items())
    if is_correct:
        return (
            f"Question: {question}\n{opts}\n\n"
            f"The correct answer is {correct_letter}. "
            f"Please explain in 2-3 sentences why this is the right answer, "
            f"focusing on the key medical concept."
        )
    else:
        return (
            f"Question: {question}\n{opts}\n\n"
            f"The correct answer is {correct_letter} (the student answered {student_letter}). "
            f"Please explain in 2-3 sentences why {correct_letter} is correct and "
            f"why {student_letter} is incorrect, focusing on the key medical concept."
        )


def _generate_explanation(event: dict) -> str:
    question       = event["question"]
    correct_letter = event["correct_answer"]
    student_letter = event.get("student_answer", "")
    is_correct     = event.get("is_correct", False)
    options = {
        "a": event["option_a"],
        "b": event["option_b"],
        "c": event["option_c"],
        "d": event["option_d"],
    }

    prompt = _build_explanation_prompt(question, options, correct_letter, is_correct, student_letter)

    try:
        result = _hf_post(HF_EXPLAIN_MODEL, {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.4,
                "do_sample": True,
                "return_full_text": False,
            },
        })
        if isinstance(result, list) and result:
            text = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            text = result.get("generated_text", "")
        else:
            text = str(result)
        return text.strip()
    except Exception as e:
        # Non-fatal: return a basic explanation rather than failing the whole pipeline
        return (
            f"The correct answer is {correct_letter}. "
            f"(Explanation generation failed: {type(e).__name__})"
        )


_LETTER_RE = re.compile(r"\b([ABCD])\b")


def _build_answer_prompt(question: str, options: dict) -> str:
    opts = "\n".join(f"{k.upper()}. {v}" for k, v in options.items())
    return (
        f"Question: {question}\n{opts}\n\n"
        f"Respond with only a single letter: A, B, C, or D.\nAnswer:"
    )


def _parse_letter(text: str) -> str | None:
    if not text:
        return None
    m = _LETTER_RE.search(text.upper())
    return m.group(1) if m else None


def _hf_answer(model_id: str, question: str, options: dict) -> str | None:
    """Ask HF model to pick A/B/C/D. Returns parsed letter or None on failure."""
    prompt = _build_answer_prompt(question, options)
    try:
        result = _hf_post(model_id, {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 4,
                "temperature": 0.0,
                "do_sample": False,
                "return_full_text": False,
            },
        }, timeout=30)
    except Exception:
        return None
    if isinstance(result, list) and result:
        first = result[0]
        # Text-generation shape
        if isinstance(first, dict) and "generated_text" in first:
            return _parse_letter(first["generated_text"])
        # Classification shape: [{label:"A",score:..},...] — pick top
        if isinstance(first, dict) and "label" in first:
            top = max(result, key=lambda x: x.get("score", 0))
            return _parse_letter(top.get("label"))
    if isinstance(result, dict) and "generated_text" in result:
        return _parse_letter(result["generated_text"])
    return None


def _demo_model_answer(model_id: str, question_id: str, correct_letter: str) -> str:
    """Deterministic canned answer so demo mode produces a stable per-model score.
    Roughly: gemma ~80% accurate, deberta ~70%, others ~60%.
    """
    targets = {"gemma": 8, "deberta": 7}
    threshold = 6
    for key, t in targets.items():
        if key in model_id.lower():
            threshold = t
            break
    seed = f"{model_id}|{question_id}"
    bucket = int(hashlib.sha256(seed.encode()).hexdigest(), 16) % 10
    if bucket < threshold:
        return correct_letter
    # pick a deterministic wrong letter
    wrongs = [l for l in "ABCD" if l != correct_letter]
    return wrongs[bucket % len(wrongs)]


def _evaluate_models(event: dict, demo: bool) -> list:
    """Run each configured eval model on this question; return list of per-model results."""
    qid = event.get("questionId", "")
    correct = event.get("correct_answer", "").upper()
    options = {
        "a": event["option_a"], "b": event["option_b"],
        "c": event["option_c"], "d": event["option_d"],
    }
    results = []
    for mid in _get_eval_models():
        if demo:
            answer = _demo_model_answer(mid, qid, correct)
        else:
            answer = _hf_answer(mid, event["question"], options)
        results.append({
            "model":       mid,
            "answer":      answer or "",
            "is_correct":  bool(answer) and answer.upper() == correct,
        })
    return results


def lambda_handler(event, context):
    demo = _is_demo_mode()
    if demo:
        # Small artificial delay (~250 ms) so the Step Functions Map fan-out
        # is still visible animating in the console during the live demo.
        time.sleep(0.25)
        explanation = _demo_explanation(event)
    else:
        explanation = _generate_explanation(event)
    model_answers = _evaluate_models(event, demo=demo)
    return {**event, "explanation": explanation, "model_answers": model_answers}
