"""
Upload Image Lambda — admin-only.
POST /upload-image
Body: {"filename": "...", "content_type": "image/png", "data_base64": "..."}
Writes to S3 under uploads/{uuid}.{ext}, returns public CloudFront URL.
"""

import base64
import json
import os
import re
import uuid

import boto3

BUCKET = os.environ["FRONTEND_BUCKET"]
PUBLIC_BASE = os.environ.get("PUBLIC_BASE", "").rstrip("/")

s3 = boto3.client("s3")

ALLOWED_TYPES = {
    "image/png":  "png",
    "image/jpeg": "jpg",
    "image/jpg":  "jpg",
    "image/webp": "webp",
    "image/gif":  "gif",
}
MAX_BYTES = 4 * 1024 * 1024  # 4 MB — stay under the 6 MB Lambda sync payload limit


def _response(status, body):
    return {"statusCode": status, "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body)}


def _is_admin(event):
    claims = (event.get("requestContext", {})
                   .get("authorizer", {}).get("jwt", {}).get("claims", {}))
    groups = claims.get("cognito:groups", "")
    if isinstance(groups, str):
        groups = groups.strip("[]").split(",")
    return "Admins" in [g.strip() for g in groups]


def lambda_handler(event, context):
    if event.get("requestContext", {}).get("http", {}).get("method") != "POST":
        return _response(405, {"error": "POST only"})
    if not _is_admin(event):
        return _response(403, {"error": "Admin access required"})

    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return _response(400, {"error": "Invalid JSON"})

    ct = (body.get("content_type") or "").lower()
    data_b64 = body.get("data_base64")
    if ct not in ALLOWED_TYPES:
        return _response(400, {"error": f"Unsupported content_type. Allowed: {sorted(ALLOWED_TYPES)}"})
    if not data_b64 or not isinstance(data_b64, str):
        return _response(400, {"error": "Missing data_base64"})

    try:
        data = base64.b64decode(data_b64, validate=True)
    except Exception:
        return _response(400, {"error": "data_base64 is not valid base64"})
    if len(data) > MAX_BYTES:
        return _response(413, {"error": f"Image exceeds {MAX_BYTES} bytes"})

    ext = ALLOWED_TYPES[ct]
    key = f"uploads/{uuid.uuid4().hex}.{ext}"

    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=data,
        ContentType=ct,
        CacheControl="public, max-age=31536000, immutable",
    )

    url = f"{PUBLIC_BASE}/{key}" if PUBLIC_BASE else f"s3://{BUCKET}/{key}"
    return _response(201, {"url": url, "key": key})
