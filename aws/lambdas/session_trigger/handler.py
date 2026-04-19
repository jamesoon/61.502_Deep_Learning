"""
Cognito Lambda Trigger
=======================
Handles two Cognito events:

1. PostConfirmation_ConfirmSignUp
   - Adds new users to the "Students" group automatically.

2. TokenGeneration_HostedAuth / TokenGeneration_RefreshTokens
   - Injects `role` claim into the ID token JWT so the frontend
     can read it from the decoded token without an extra API call.
   - role = "Admin" if user is in Admins group, else "Student"
"""

import json
import os
import boto3

USER_POOL_ID = os.environ.get("USER_POOL_ID", "")
REGION       = os.environ.get("REGION", "ap-southeast-1")

cognito = boto3.client("cognito-idp", region_name=REGION)


def _get_user_groups(user_pool_id: str, username: str) -> list[str]:
    try:
        resp = cognito.admin_list_groups_for_user(
            UserPoolId=user_pool_id,
            Username=username,
        )
        return [g["GroupName"] for g in resp.get("Groups", [])]
    except Exception:
        return []


def lambda_handler(event, context):
    trigger_source = event.get("triggerSource", "")
    user_pool_id   = event.get("userPoolId", USER_POOL_ID)
    username       = event.get("userName", "")

    # ── PostConfirmation: add new user to Students group ──────────────────────
    if trigger_source == "PostConfirmation_ConfirmSignUp":
        try:
            cognito.admin_add_user_to_group(
                UserPoolId=user_pool_id,
                Username=username,
                GroupName="Students",
            )
        except Exception as e:
            # Non-fatal: user can still log in, just won't have group
            print(f"[WARNING] Failed to add {username} to Students group: {e}")
        return event

    # ── Pre-Token Generation: inject role claim ───────────────────────────────
    if trigger_source.startswith("TokenGeneration"):
        groups = _get_user_groups(user_pool_id, username)
        role = "Admin" if "Admins" in groups else "Student"

        # Inject into ID token claims
        event.setdefault("response", {})
        event["response"]["claimsOverrideDetails"] = {
            "claimsToAddOrOverride": {
                "custom:role": role,
            },
        }
        return event

    # Unknown trigger — pass through unchanged
    return event
