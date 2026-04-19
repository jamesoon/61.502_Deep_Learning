#!/usr/bin/env bash
#
# Fast cleanup of RETAIN-policy orphans left behind by a failed cdk deploy.
#
# WHY THIS EXISTS:
#   When cdk deploy fails partway through and CloudFormation rolls back,
#   resources declared with RemovalPolicy.RETAIN survive the rollback by design
#   (this protects production data). On a fresh re-deploy attempt, CFN then
#   errors with:
#     "Resource of type 'AWS::DynamoDB::Table' with identifier
#      'medmcqa_questions' already exists."
#
# WHAT THIS SCRIPT DOES:
#   - Deletes the three medmcqa DDB tables IFF they're empty (safety check)
#   - Deletes the medmcqa-users Cognito pool IFF it has no confirmed users
#   - Deletes the orphan frontend S3 bucket if any
#   - Deletes the /medmcqa/demo_mode SSM parameter if it survived
#
# It does NOT touch:
#   - CDKToolkit bootstrap stacks
#   - Route 53 hosted zone
#   - SSM /medmcqa/hf_token (preserved by default)
#
# Use this INSTEAD of cdk_teardown.sh when the stack is already gone
# (ROLLBACK_COMPLETE → CFN auto-deletes the stack record). This script skips
# the venv setup and cdk destroy round-trip — runs in ~30 seconds.
#
# Flags:
#   --yes       Skip confirmation prompts
#   --force     Delete tables/pool even if they contain data (use with care)
#
# Usage:
#   bash aws/cleanup_retain_orphans.sh
#   bash aws/cleanup_retain_orphans.sh --yes
#   bash aws/cleanup_retain_orphans.sh --yes --force   # delete even if non-empty

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$SCRIPT_DIR/retain_cleanup.log"
: > "$LOG"

YES=0
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --yes|-y)   YES=1 ;;
        --force)    FORCE=1 ;;
        -h|--help)
            grep -E "^#( |$)" "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown flag: $arg (use --help)"; exit 2 ;;
    esac
done

log()  { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
fail() { log "FATAL: $*"; exit 1; }
warn() { log "WARN: $*"; }
try()  { log "+ $*"; eval "$@" 2>&1 | tee -a "$LOG"; local rc=${PIPESTATUS[0]}; [[ $rc -ne 0 ]] && warn "(exit $rc — continuing)"; return 0; }

# ── Load creds ────────────────────────────────────────────────────────────────
ENV_FILE="$SCRIPT_DIR/cdk/.env"
[[ -f "$ENV_FILE" ]] || fail ".env not found at $ENV_FILE"
set -a; source "$ENV_FILE"; set +a
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-ap-southeast-1}"

aws sts get-caller-identity >>"$LOG" 2>&1 || fail "STS get-caller-identity failed"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
log "=== RETAIN-orphan cleanup ==="
log "Account: $ACCOUNT  Region: $AWS_DEFAULT_REGION  Force: $FORCE"

# ── Inventory ──────────────────────────────────────────────────────────────────
log ""
log "--- Inventory ---"

DDB_FOUND=()
for tbl in medmcqa_questions medmcqa_submissions medmcqa_user_sessions; do
    STATUS=$(aws dynamodb describe-table --table-name "$tbl" \
        --query 'Table.TableStatus' --output text 2>/dev/null || echo "MISSING")
    if [[ "$STATUS" != "MISSING" ]]; then
        ITEMS=$(aws dynamodb scan --table-name "$tbl" --select COUNT \
            --query 'Count' --output text 2>/dev/null || echo "?")
        log "  DDB $tbl: $STATUS ($ITEMS items)"
        DDB_FOUND+=("$tbl:$ITEMS")
    else
        log "  DDB $tbl: not present"
    fi
done

USER_POOL_ID=$(aws cognito-idp list-user-pools --max-results 60 \
    --query "UserPools[?Name=='medmcqa-users'].Id | [0]" --output text 2>/dev/null || echo "None")
USER_COUNT=0
if [[ "$USER_POOL_ID" != "None" && -n "$USER_POOL_ID" ]]; then
    USER_COUNT=$(aws cognito-idp list-users --user-pool-id "$USER_POOL_ID" \
        --query 'length(Users)' --output text 2>/dev/null || echo "?")
    log "  Cognito medmcqa-users: $USER_POOL_ID ($USER_COUNT users)"
else
    log "  Cognito medmcqa-users: not present"
fi

FRONTEND_BUCKET="medmcqa-frontend-$ACCOUNT"
BUCKET_PRESENT=0
if aws s3api head-bucket --bucket "$FRONTEND_BUCKET" 2>/dev/null; then
    OBJ_COUNT=$(aws s3 ls "s3://$FRONTEND_BUCKET" --recursive --summarize 2>/dev/null | grep "Total Objects" | awk '{print $3}' || echo "?")
    log "  S3 $FRONTEND_BUCKET: present ($OBJ_COUNT objects)"
    BUCKET_PRESENT=1
else
    log "  S3 $FRONTEND_BUCKET: not present"
fi

DEMO_PARAM=$(aws ssm get-parameter --name /medmcqa/demo_mode --region "$AWS_DEFAULT_REGION" \
    --query 'Parameter.Value' --output text 2>/dev/null || echo "MISSING")
log "  SSM /medmcqa/demo_mode: $DEMO_PARAM"

# ── Safety check ──────────────────────────────────────────────────────────────
if [[ ${#DDB_FOUND[@]} -eq 0 && "$USER_POOL_ID" == "None" && $BUCKET_PRESENT -eq 0 && "$DEMO_PARAM" == "MISSING" ]]; then
    log ""
    log "Nothing to clean up — environment is already pristine. You can run cdk_deploy.sh."
    exit 0
fi

# Block on non-empty tables/pool unless --force
if [[ $FORCE -eq 0 ]]; then
    BLOCK=0
    for entry in "${DDB_FOUND[@]}"; do
        items="${entry##*:}"
        [[ "$items" =~ ^[0-9]+$ && "$items" -gt 0 ]] && { warn "Table ${entry%%:*} has $items items — pass --force to delete anyway"; BLOCK=1; }
    done
    if [[ "$USER_COUNT" =~ ^[0-9]+$ && "$USER_COUNT" -gt 0 ]]; then
        warn "Cognito pool has $USER_COUNT users — pass --force to delete anyway"
        BLOCK=1
    fi
    if [[ $BLOCK -eq 1 ]]; then
        log ""
        log "Aborted: data present. Re-run with --force or use cdk_teardown.sh --keep-data --keep-users to preserve."
        exit 1
    fi
fi

# ── Confirm ────────────────────────────────────────────────────────────────────
log ""
if [[ $YES -eq 0 ]]; then
    read -r -p "Proceed with cleanup? Type 'cleanup' to confirm: " confirm
    [[ "$confirm" == "cleanup" ]] || { log "Aborted by user."; exit 0; }
fi

# ── 1. DDB tables ──────────────────────────────────────────────────────────────
log ""
log "--- Deleting DDB tables ---"
for entry in "${DDB_FOUND[@]}"; do
    tbl="${entry%%:*}"
    try "aws dynamodb delete-table --table-name $tbl --query 'TableDescription.TableStatus' --output text"
done
# Wait for deletes
log "Waiting for tables to disappear..."
for entry in "${DDB_FOUND[@]}"; do
    tbl="${entry%%:*}"
    for i in {1..30}; do
        STATUS=$(aws dynamodb describe-table --table-name "$tbl" \
            --query 'Table.TableStatus' --output text 2>/dev/null || echo "DELETED")
        [[ "$STATUS" == "DELETED" ]] && { log "  $tbl: deleted ✓"; break; }
        sleep 2
    done
done

# ── 2. Cognito user pool ───────────────────────────────────────────────────────
if [[ "$USER_POOL_ID" != "None" && -n "$USER_POOL_ID" ]]; then
    log ""
    log "--- Deleting Cognito user pool $USER_POOL_ID ---"
    DOMAIN=$(aws cognito-idp describe-user-pool --user-pool-id "$USER_POOL_ID" \
        --query 'UserPool.Domain' --output text 2>/dev/null || echo "None")
    if [[ "$DOMAIN" != "None" && -n "$DOMAIN" ]]; then
        try "aws cognito-idp delete-user-pool-domain --user-pool-id '$USER_POOL_ID' --domain '$DOMAIN'"
    fi
    try "aws cognito-idp delete-user-pool --user-pool-id '$USER_POOL_ID'"
    VERIFY=$(aws cognito-idp describe-user-pool --user-pool-id "$USER_POOL_ID" \
        --query 'UserPool.Id' --output text 2>/dev/null || echo "MISSING")
    [[ "$VERIFY" == "MISSING" ]] && log "  Cognito pool deleted ✓" || warn "Cognito pool may still exist"
fi

# ── 3. S3 frontend bucket ──────────────────────────────────────────────────────
if [[ $BUCKET_PRESENT -eq 1 ]]; then
    log ""
    log "--- Emptying & deleting S3 bucket $FRONTEND_BUCKET ---"
    try "aws s3 rm s3://$FRONTEND_BUCKET --recursive"
    # Versioned objects
    try "aws s3api delete-objects --bucket '$FRONTEND_BUCKET' \
        --delete \"\$(aws s3api list-object-versions --bucket '$FRONTEND_BUCKET' \
            --output json --query '{Objects: Versions[].{Key:Key,VersionId:VersionId}}' 2>/dev/null)\" >/dev/null 2>&1"
    try "aws s3api delete-objects --bucket '$FRONTEND_BUCKET' \
        --delete \"\$(aws s3api list-object-versions --bucket '$FRONTEND_BUCKET' \
            --output json --query '{Objects: DeleteMarkers[].{Key:Key,VersionId:VersionId}}' 2>/dev/null)\" >/dev/null 2>&1"
    try "aws s3api delete-bucket --bucket $FRONTEND_BUCKET"
fi

# ── 4. SSM demo_mode ───────────────────────────────────────────────────────────
if [[ "$DEMO_PARAM" != "MISSING" ]]; then
    log ""
    log "--- Deleting SSM /medmcqa/demo_mode ---"
    try "aws ssm delete-parameter --name /medmcqa/demo_mode --region $AWS_DEFAULT_REGION"
fi

# ── 5. Final state ─────────────────────────────────────────────────────────────
log ""
log "=== FINAL STATE ==="
for tbl in medmcqa_questions medmcqa_submissions medmcqa_user_sessions; do
    STATUS=$(aws dynamodb describe-table --table-name "$tbl" \
        --query 'Table.TableStatus' --output text 2>/dev/null || echo "MISSING")
    log "  DDB $tbl: $STATUS"
done
USER_POOL_ID=$(aws cognito-idp list-user-pools --max-results 60 \
    --query "UserPools[?Name=='medmcqa-users'].Id | [0]" --output text 2>/dev/null || echo "None")
log "  Cognito medmcqa-users: $USER_POOL_ID"
log ""
log "Next: bash aws/cdk_deploy.sh"
log "Full log: $LOG"
