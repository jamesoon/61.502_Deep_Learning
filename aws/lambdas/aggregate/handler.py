"""
Aggregate Lambda — Step Functions final state
=============================================
Input (from Step Functions after Map completes):
{
  "submissionId": str,
  "gradedQuestions": [
    { ...question fields..., "explanation": str },
    ...
  ]
}

Merges explanations into DDB submission and marks status=COMPLETE.
"""

import json
import os
from datetime import datetime, timezone

import boto3

SUBMISSIONS_TABLE = os.environ["SUBMISSIONS_TABLE"]
ddb = boto3.resource("dynamodb")
submissions_table = ddb.Table(SUBMISSIONS_TABLE)


def _model_short_name(model_id: str) -> str:
    """'jamezoon/gemma-3-4b-it-medmcqa-lora' → 'gemma-3-4b-it-medmcqa-lora'."""
    return model_id.split("/", 1)[-1] if "/" in model_id else model_id


def lambda_handler(event, context):
    submission_id    = event["submissionId"]
    graded_questions = event.get("gradedQuestions", [])

    # Build explanation map {questionId: explanation}
    explanations = {
        q["questionId"]: q.get("explanation", "")
        for q in graded_questions
    }

    # Roll up per-model correctness across the graded questions.
    # tallies: {model_id: {"correct": int, "total": int}}
    tallies: dict = {}
    for gq in graded_questions:
        for m in gq.get("model_answers") or []:
            mid = m.get("model", "")
            if not mid:
                continue
            t = tallies.setdefault(mid, {"correct": 0, "total": 0})
            t["total"] += 1
            if m.get("is_correct"):
                t["correct"] += 1

    model_scores = [
        {
            "model":       mid,
            "short_name":  _model_short_name(mid),
            "correct":     t["correct"],
            "total":       t["total"],
            "accuracy_pct": round(100 * t["correct"] / t["total"]) if t["total"] else 0,
        }
        for mid, t in tallies.items()
    ]
    model_scores.sort(key=lambda x: x["accuracy_pct"], reverse=True)

    # Fetch current submission
    resp = submissions_table.get_item(Key={"submissionId": submission_id})
    submission = resp.get("Item")
    if not submission:
        raise ValueError(f"Submission {submission_id} not found")

    # Merge explanations into results
    updated_results = []
    for r in submission.get("results", []):
        r = dict(r)
        r["explanation"] = explanations.get(r["questionId"], "")
        updated_results.append(r)

    # Update submission to COMPLETE
    submissions_table.update_item(
        Key={"submissionId": submission_id},
        UpdateExpression=(
            "SET #status = :s, #results = :r, #gradedAt = :g, "
            "#model_scores = :ms"
        ),
        ExpressionAttributeNames={
            "#status":       "status",
            "#results":      "results",
            "#gradedAt":     "gradedAt",
            "#model_scores": "model_scores",
        },
        ExpressionAttributeValues={
            ":s":  "COMPLETE",
            ":r":  updated_results,
            ":g":  datetime.now(timezone.utc).isoformat(),
            ":ms": model_scores,
        },
    )

    return {
        "submissionId": submission_id,
        "status":       "COMPLETE",
        "total":        len(updated_results),
        "model_scores": model_scores,
    }
