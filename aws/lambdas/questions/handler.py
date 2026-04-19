"""
Questions Lambda — CRUD for MCQ questions.
Routes (all require Cognito JWT):
  GET    /questions              → list all (any authenticated user)
  POST   /questions              → create (Admins only)
  PUT    /questions/{questionId} → update (Admins only)
  DELETE /questions/{questionId} → delete (Admins only)
"""

import json
import os
import uuid
from datetime import datetime, timezone

import boto3
from boto3.dynamodb.conditions import Attr

TABLE_NAME = os.environ["QUESTIONS_TABLE"]
ddb = boto3.resource("dynamodb")
table = ddb.Table(TABLE_NAME)


def _response(status: int, body):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _is_admin(event: dict) -> bool:
    """Check if the caller belongs to the Admins Cognito group."""
    claims = (event.get("requestContext", {})
                   .get("authorizer", {})
                   .get("jwt", {})
                   .get("claims", {}))
    groups = claims.get("cognito:groups", "")
    if isinstance(groups, str):
        groups = groups.strip("[]").split(",")
    return "Admins" in [g.strip() for g in groups]


def _get_caller_id(event: dict) -> str:
    claims = (event.get("requestContext", {})
                   .get("authorizer", {})
                   .get("jwt", {})
                   .get("claims", {}))
    return claims.get("sub", "unknown")


def _list_questions():
    """Scan all questions (table is small for a course project)."""
    resp = table.scan()
    items = resp.get("Items", [])
    while "LastEvaluatedKey" in resp:
        resp = table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"])
        items.extend(resp.get("Items", []))
    # Sort by createdAt descending
    items.sort(key=lambda x: x.get("createdAt", ""), reverse=True)
    return items


def lambda_handler(event, context):
    method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    path   = event.get("rawPath", "")
    params = event.get("pathParameters") or {}
    question_id = params.get("questionId")

    # ── GET /questions ────────────────────────────────────────────────────────
    if method == "GET" and not question_id:
        items = _list_questions()
        return _response(200, items)

    # ── GET /questions/{questionId} ───────────────────────────────────────────
    if method == "GET" and question_id:
        resp = table.get_item(Key={"questionId": question_id})
        item = resp.get("Item")
        if not item:
            return _response(404, {"error": "Question not found"})
        return _response(200, item)

    # ── Admin-only below ──────────────────────────────────────────────────────
    if not _is_admin(event):
        return _response(403, {"error": "Admin access required"})

    body = {}
    if event.get("body"):
        body = json.loads(event["body"])

    # ── POST /questions ───────────────────────────────────────────────────────
    if method == "POST":
        required = ["question", "option_a", "option_b", "option_c", "option_d", "correct_answer"]
        missing = [f for f in required if not body.get(f)]
        if missing:
            return _response(400, {"error": f"Missing fields: {missing}"})

        item = {
            "questionId":     str(uuid.uuid4()),
            "question":       body["question"],
            "option_a":       body["option_a"],
            "option_b":       body["option_b"],
            "option_c":       body["option_c"],
            "option_d":       body["option_d"],
            "correct_answer": body["correct_answer"].upper(),
            "subject":        body.get("subject", ""),
            "topic":          body.get("topic", ""),
            "explanation":    body.get("explanation", ""),
            "image_url":      body.get("image_url", ""),
            "createdBy":      _get_caller_id(event),
            "createdAt":      datetime.now(timezone.utc).isoformat(),
        }
        table.put_item(Item=item)
        return _response(201, item)

    # ── PUT /questions/{questionId} ───────────────────────────────────────────
    if method == "PUT" and question_id:
        # Build update expression from provided fields
        updatable = ["question", "option_a", "option_b", "option_c", "option_d",
                     "correct_answer", "subject", "topic", "explanation", "image_url"]
        updates = {k: v for k, v in body.items() if k in updatable}
        if not updates:
            return _response(400, {"error": "No updatable fields provided"})
        if "correct_answer" in updates:
            updates["correct_answer"] = updates["correct_answer"].upper()

        updates["updatedAt"] = datetime.now(timezone.utc).isoformat()
        expr = "SET " + ", ".join(f"#{k} = :{k}" for k in updates)
        names = {f"#{k}": k for k in updates}
        values = {f":{k}": v for k, v in updates.items()}

        try:
            resp = table.update_item(
                Key={"questionId": question_id},
                UpdateExpression=expr,
                ExpressionAttributeNames=names,
                ExpressionAttributeValues=values,
                ConditionExpression=Attr("questionId").exists(),
                ReturnValues="ALL_NEW",
            )
        except ddb.meta.client.exceptions.ConditionalCheckFailedException:
            return _response(404, {"error": "Question not found"})
        return _response(200, resp["Attributes"])

    # ── DELETE /questions/{questionId} ────────────────────────────────────────
    if method == "DELETE" and question_id:
        try:
            table.delete_item(
                Key={"questionId": question_id},
                ConditionExpression=Attr("questionId").exists(),
            )
        except ddb.meta.client.exceptions.ConditionalCheckFailedException:
            return _response(404, {"error": "Question not found"})
        return _response(200, {"deleted": question_id})

    return _response(405, {"error": f"Method {method} not allowed"})
