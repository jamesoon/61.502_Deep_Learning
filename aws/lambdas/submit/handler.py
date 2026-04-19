"""
Submit Lambda
=============
POST /submit
  Body: { answers: {questionId: "A"|"B"|"C"|"D", ...} }
  - Loads questions from DDB
  - Computes is_correct for each answered question immediately
  - Saves submission to DDB with status=GRADING, partial results
  - Starts Step Functions async execution for LLM explanation generation
  - Returns {submissionId, status, score_percent, correct, total} immediately

GET /submissions/{submissionId}
  - Returns submission status + results (poll until status=COMPLETE)
"""

import json
import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal

import boto3


class _DecimalEncoder(json.JSONEncoder):
    """DDB returns numerics as Decimal — coerce to int/float for JSON output."""
    def default(self, o):
        if isinstance(o, Decimal):
            return int(o) if o == o.to_integral_value() else float(o)
        return super().default(o)

QUESTIONS_TABLE   = os.environ["QUESTIONS_TABLE"]
SUBMISSIONS_TABLE = os.environ["SUBMISSIONS_TABLE"]
STATE_MACHINE_ARN = os.environ["STATE_MACHINE_ARN"]

ddb = boto3.resource("dynamodb")
questions_table   = ddb.Table(QUESTIONS_TABLE)
submissions_table = ddb.Table(SUBMISSIONS_TABLE)
sf_client         = boto3.client("stepfunctions")


def _response(status: int, body):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body, cls=_DecimalEncoder),
    }


def _get_caller(event: dict) -> tuple[str, str]:
    """Returns (userId, studentName) from JWT claims."""
    claims = (event.get("requestContext", {})
                   .get("authorizer", {})
                   .get("jwt", {})
                   .get("claims", {}))
    user_id = claims.get("sub", "anonymous")
    name    = claims.get("name") or claims.get("email", "Student")
    return user_id, name


def _fetch_all_questions() -> dict:
    """Returns {questionId: item} map."""
    resp = questions_table.scan()
    items = resp.get("Items", [])
    while "LastEvaluatedKey" in resp:
        resp = questions_table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"])
        items.extend(resp.get("Items", []))
    return {q["questionId"]: q for q in items}


def _handle_submit(event: dict):
    body = {}
    if event.get("body"):
        body = json.loads(event["body"])

    answers = body.get("answers", {})  # {questionId: "A"|"B"|"C"|"D"}
    if not answers:
        return _response(400, {"error": "answers map is required"})

    user_id, student_name = _get_caller(event)
    all_questions = _fetch_all_questions()

    # Score immediately — no ML needed for correctness check
    partial_results = []
    correct = 0
    for qid, student_answer in answers.items():
        q = all_questions.get(qid)
        if not q:
            continue
        is_correct = q.get("correct_answer", "").upper() == (student_answer or "").upper()
        if is_correct:
            correct += 1
        partial_results.append({
            "questionId":     qid,
            "question":       q.get("question", ""),
            "option_a":       q.get("option_a", ""),
            "option_b":       q.get("option_b", ""),
            "option_c":       q.get("option_c", ""),
            "option_d":       q.get("option_d", ""),
            "correct_answer": q.get("correct_answer", ""),
            "student_answer": student_answer,
            "is_correct":     is_correct,
            "image_url":      q.get("image_url", ""),
            "explanation":    None,   # filled in by Step Functions
        })

    total = len(partial_results)
    score_pct = round((correct / total * 100)) if total else 0
    submission_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    submission = {
        "submissionId":    submission_id,
        "userId":          user_id,
        "studentName":     student_name,
        "status":          "GRADING",
        "score_percent":   score_pct,
        "correct":         correct,
        "total_questions": total,
        "results":         partial_results,
        "submittedAt":     now,
        "gradedAt":        None,
    }
    submissions_table.put_item(Item=submission)

    # Start async Step Functions execution for explanation generation
    # Pass questions that need explanations
    sf_input = {
        "submissionId": submission_id,
        "questions": [
            {
                "submissionId":  submission_id,
                "questionId":    r["questionId"],
                "question":      r["question"],
                "option_a":      r["option_a"],
                "option_b":      r["option_b"],
                "option_c":      r["option_c"],
                "option_d":      r["option_d"],
                "correct_answer": r["correct_answer"],
                "student_answer": r["student_answer"],
                "is_correct":    r["is_correct"],
            }
            for r in partial_results
        ],
    }

    sf_client.start_execution(
        stateMachineArn=STATE_MACHINE_ARN,
        name=submission_id,
        input=json.dumps(sf_input),
    )

    return _response(202, {
        "submissionId":    submission_id,
        "status":          "GRADING",
        "score_percent":   score_pct,
        "correct":         correct,
        "total_questions": total,
        "student_name":    student_name,
        "results":         partial_results,
    })


def _handle_get_submission(event: dict, submission_id: str):
    resp = submissions_table.get_item(Key={"submissionId": submission_id})
    item = resp.get("Item")
    if not item:
        return _response(404, {"error": "Submission not found"})

    # Only allow the owner to see their submission
    user_id, _ = _get_caller(event)
    if item.get("userId") != user_id:
        return _response(403, {"error": "Access denied"})

    return _response(200, item)


# Fixed default exam. Exam Groups admin UI isn't wired to DDB yet, so every
# student gets the same synthetic "MedMCQA Demo Exam" pulled from the questions
# table. `EXAM_QUESTION_COUNT` caps the number of questions per attempt.
EXAM_QUESTION_COUNT = int(os.environ.get("EXAM_QUESTION_COUNT", "5"))
DEFAULT_GROUP_ID    = "default"


def _pick_exam_questions():
    """Fresh random subset on every call so each dashboard visit / Start Exam
    gives a different sample. No assignment table — this is the demo-default path."""
    import random
    items = list(_fetch_all_questions().values())
    if not items:
        return []
    return random.sample(items, k=min(EXAM_QUESTION_COUNT, len(items)))


def _handle_my_exam(event: dict):
    """Synthetic default exam — see note above."""
    qs = _pick_exam_questions()
    if not qs:
        return _response(200, {"assigned": False})
    return _response(200, {
        "assigned":         True,
        "groupId":          DEFAULT_GROUP_ID,
        "name":             "MedMCQA Demo Exam",
        "description":      "Randomly sampled from the seeded MedMCQA + MCAT question bank.",
        "duration_minutes": 20,
        "topics":           sorted({q.get("subject","") for q in qs if q.get("subject")}),
        "questions":        qs,
    })


def _handle_my_submissions(event: dict):
    """Return current user's submissions (lightweight — results array stripped)."""
    user_id, _ = _get_caller(event)
    resp = submissions_table.scan(
        FilterExpression="userId = :uid",
        ExpressionAttributeValues={":uid": user_id},
    )
    items = resp.get("Items", [])
    while "LastEvaluatedKey" in resp:
        resp = submissions_table.scan(
            FilterExpression="userId = :uid",
            ExpressionAttributeValues={":uid": user_id},
            ExclusiveStartKey=resp["LastEvaluatedKey"],
        )
        items.extend(resp.get("Items", []))
    items.sort(key=lambda x: x.get("submittedAt", ""), reverse=True)
    # Strip the bulky results array — dashboard doesn't need per-Q detail here.
    trimmed = [{k: v for k, v in it.items() if k != "results"} for it in items]
    return _response(200, trimmed)


def lambda_handler(event, context):
    method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    path   = event.get("rawPath", "") or ""
    params = event.get("pathParameters") or {}
    submission_id = params.get("submissionId")

    if method == "POST" and path.rstrip("/").endswith("/submit"):
        return _handle_submit(event)

    if method == "GET" and submission_id:
        return _handle_get_submission(event, submission_id)

    if method == "GET" and path.rstrip("/").endswith("/my-exam"):
        return _handle_my_exam(event)

    if method == "GET" and path.rstrip("/").endswith("/my-submissions"):
        return _handle_my_submissions(event)

    return _response(405, {"error": f"Method {method} {path} not allowed"})
