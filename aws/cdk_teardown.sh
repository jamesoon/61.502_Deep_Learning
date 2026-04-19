#!/usr/bin/env bash
#
# Full teardown of MedMCQAStack and all side-effect resources.
#
# What `cdk destroy` handles automatically:
#   - CloudFront distribution
#   - Route 53 alias record (dl.mdaie-sutd.fit)
#   - ACM cert in us-east-1 (via custom resource)
#   - API Gateway HTTP API + integrations + authorizer
#   - Lambda functions + their auto-created log groups
#   - Step Functions state machine + its role
#   - IAM roles (LambdaRole, SubmitRole)
#   - S3 frontend bucket (auto_delete_objects=True handles objects)
#   - SSM parameter /medmcqa/demo_mode
#
# What `cdk destroy` does NOT handle (RemovalPolicy.RETAIN) — this script
# deletes them explicitly:
#   - DynamoDB: medmcqa_questions, medmcqa_submissions, medmcqa_user_sessions
#   - Cognito: medmcqa-users user pool
#
# What's intentionally left behind (reuse for next deploy):
#   - CDKToolkit bootstrap stacks in ap-southeast-1 and us-east-1
#   - Route 53 hosted zone (you own the domain)
#   - Optional SSM /medmcqa/hf_token (delete with --purge-ssm)
#
# Flags:
#   --yes         Skip confirmation prompts
#   --purge-ssm   Also delete /medmcqa/hf_token and any stragglers under /medmcqa/
#   --keep-data   Skip DDB table deletion (preserves submissions history)
#   --keep-users  Skip Cognito user pool deletion (preserves user accounts)
#
# Usage:
#   bash aws/cdk_teardown.sh              # interactive, full teardown
#   bash aws/cdk_teardown.sh --yes        # non-interactive, full teardown
#   bash aws/cdk_teardown.sh --keep-data --keep-users   # just kill the app
#
# All output → aws/cdk_teardown.log

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$SCRIPT_DIR/cdk_teardown.log"
: > "$LOG"

# ── Flag parsing ───────────────────────────────────────────────────────────────
YES=0
PURGE_SSM=0
KEEP_DATA=0
KEEP_USERS=0
for arg in "$@"; do
    case "$arg" in
        --yes|-y)      YES=1 ;;
        --purge-ssm)   PURGE_SSM=1 ;;
        --keep-data)   KEEP_DATA=1 ;;
        --keep-users)  KEEP_USERS=1 ;;
        -h|--help)
            grep -E "^#( |$)" "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown flag: $arg (use --help)"; exit 2 ;;
    esac
done

log()  { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
fail() { log "FATAL: $*"; log "See full log at $LOG"; exit 1; }
warn() { log "WARN: $*"; }
run()  { log "+ $*"; eval "$@" 2>&1 | tee -a "$LOG"; return ${PIPESTATUS[0]}; }
# Like run() but non-fatal — teardown is best-effort
try()  { log "+ $*"; eval "$@" 2>&1 | tee -a "$LOG"; local rc=${PIPESTATUS[0]}; [[ $rc -ne 0 ]] && warn "(exit $rc — continuing)"; return 0; }

log "=== MedMCQAStack teardown ==="
log "Script dir: $SCRIPT_DIR"
log "Log: $LOG"
log "Flags: yes=$YES purge_ssm=$PURGE_SSM keep_data=$KEEP_DATA keep_users=$KEEP_USERS"

# ── 1. Load creds ──────────────────────────────────────────────────────────────
ENV_FILE="$SCRIPT_DIR/cdk/.env"
[[ -f "$ENV_FILE" ]] || fail ".env not found at $ENV_FILE"
log "Sourcing $ENV_FILE"
set -a; source "$ENV_FILE"; set +a
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-ap-southeast-1}"
export CDK_DEFAULT_REGION="${CDK_DEFAULT_REGION:-$AWS_DEFAULT_REGION}"
export CDK_DEFAULT_ACCOUNT="${CDK_DEFAULT_ACCOUNT:-}"
log "Primary region: $AWS_DEFAULT_REGION"

# ── 2. Identity ────────────────────────────────────────────────────────────────
command -v aws     >/dev/null 2>&1 || fail "'aws' CLI missing"
command -v python3 >/dev/null 2>&1 || fail "'python3' missing"
command -v npx     >/dev/null 2>&1 || fail "'npx' missing"

if ! aws sts get-caller-identity >>"$LOG" 2>&1; then
    fail "STS get-caller-identity failed — creds invalid"
fi
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
[[ -z "$CDK_DEFAULT_ACCOUNT" ]] && export CDK_DEFAULT_ACCOUNT="$ACCOUNT"
log "Account: $ACCOUNT"

# ── 3. Safety check — show what will be destroyed ──────────────────────────────
log ""
log "=== Resources that will be DESTROYED ==="
log "  • CloudFormation stack: MedMCQAStack (ap-southeast-1)"
log "  • CloudFront distribution → https://dl.mdaie-sutd.fit"
log "  • Route 53 record: dl.mdaie-sutd.fit"
log "  • ACM certificate for dl.mdaie-sutd.fit (us-east-1)"
log "  • API Gateway HTTP API, all Lambdas, Step Functions"
log "  • S3 frontend bucket: medmcqa-frontend-$ACCOUNT"
log "  • SSM parameter: /medmcqa/demo_mode"
if [[ $KEEP_DATA -eq 0 ]]; then
    log "  • DynamoDB tables: medmcqa_questions, medmcqa_submissions, medmcqa_user_sessions"
else
    log "  • (DDB tables PRESERVED per --keep-data)"
fi
if [[ $KEEP_USERS -eq 0 ]]; then
    log "  • Cognito User Pool: medmcqa-users"
else
    log "  • (Cognito user pool PRESERVED per --keep-users)"
fi
if [[ $PURGE_SSM -eq 1 ]]; then
    log "  • All /medmcqa/* SSM parameters (including /medmcqa/hf_token)"
fi
log ""
log "=== Resources that will be PRESERVED ==="
log "  • CDKToolkit bootstrap stacks (for future redeploys)"
log "  • Route 53 hosted zone for mdaie-sutd.fit"
log "  • /medmcqa/hf_token (unless --purge-ssm)"
log ""

if [[ $YES -eq 0 ]]; then
    read -r -p "Type 'destroy' to proceed: " confirmation
    if [[ "$confirmation" != "destroy" ]]; then
        log "Aborted by user."
        exit 0
    fi
fi

# ── 4. Pre-clean: empty the frontend S3 bucket (speeds up cdk destroy) ─────────
log ""
log "--- Emptying frontend S3 bucket ---"
FRONTEND_BUCKET="medmcqa-frontend-$ACCOUNT"
if aws s3api head-bucket --bucket "$FRONTEND_BUCKET" 2>/dev/null; then
    # First remove any object versions (bucket may have versioning from prior runs)
    try "aws s3api delete-objects --bucket '$FRONTEND_BUCKET' \
        --delete \"\$(aws s3api list-object-versions --bucket '$FRONTEND_BUCKET' \
            --output json --query '{Objects: Versions[].{Key:Key,VersionId:VersionId}}' 2>/dev/null)\" >/dev/null 2>&1"
    try "aws s3api delete-objects --bucket '$FRONTEND_BUCKET' \
        --delete \"\$(aws s3api list-object-versions --bucket '$FRONTEND_BUCKET' \
            --output json --query '{Objects: DeleteMarkers[].{Key:Key,VersionId:VersionId}}' 2>/dev/null)\" >/dev/null 2>&1"
    # Plain object delete as fallback
    try "aws s3 rm s3://$FRONTEND_BUCKET --recursive"
    log "Bucket emptied: $FRONTEND_BUCKET"
else
    log "Frontend bucket $FRONTEND_BUCKET not found (already deleted or never created) — skipping"
fi

# ── 5. CDK destroy ─────────────────────────────────────────────────────────────
log ""
log "--- Python venv setup ---"
CDK_DIR="$SCRIPT_DIR/cdk"
cd "$CDK_DIR" || fail "cd $CDK_DIR failed"
if [[ ! -d ".venv" ]]; then
    log "venv not found — creating"
    run "python3 -m venv .venv" || fail "venv creation failed"
fi
# shellcheck source=/dev/null
source .venv/bin/activate
export VIRTUAL_ENV="$CDK_DIR/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
run "pip install --quiet --upgrade pip" || warn "pip upgrade failed"
run "pip install --quiet -r requirements.txt" || warn "pip install requirements failed"
if ! .venv/bin/python3 -c "import aws_cdk" >>"$LOG" 2>&1; then
    fail "venv python3 cannot import aws_cdk — pip install failed"
fi

CDK="npx --yes aws-cdk@2"
log "CDK: $CDK"

log ""
log "--- cdk destroy (this can take 10-20 min — CloudFront propagation is slow) ---"
DESTROY_START=$(date +%s)
# --force skips the CDK's own interactive prompt (we already confirmed above)
if ! run "$CDK destroy --force MedMCQAStack"; then
    warn "cdk destroy reported errors — will continue with explicit cleanup"
fi
DESTROY_END=$(date +%s)
log "cdk destroy time: $(( (DESTROY_END - DESTROY_START) / 60 )) min $(( (DESTROY_END - DESTROY_START) % 60 )) sec"

# Verify stack gone
STACK_STATUS=$(aws cloudformation describe-stacks --stack-name MedMCQAStack \
    --region "$AWS_DEFAULT_REGION" --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo "MISSING")
if [[ "$STACK_STATUS" == "MISSING" ]]; then
    log "MedMCQAStack: fully deleted ✓"
elif [[ "$STACK_STATUS" == "DELETE_COMPLETE" ]]; then
    log "MedMCQAStack: DELETE_COMPLETE ✓"
else
    warn "MedMCQAStack still present with status: $STACK_STATUS (check console)"
fi

# ── 6. Delete RETAIN-policy resources ──────────────────────────────────────────
log ""
log "=== Explicit cleanup of RETAIN resources ==="

# 6a. DynamoDB tables
if [[ $KEEP_DATA -eq 0 ]]; then
    log ""
    log "--- DynamoDB tables ---"
    for tbl in medmcqa_questions medmcqa_submissions medmcqa_user_sessions; do
        STATUS=$(aws dynamodb describe-table --table-name "$tbl" \
            --query 'Table.TableStatus' --output text 2>/dev/null || echo "MISSING")
        if [[ "$STATUS" == "MISSING" ]]; then
            log "  $tbl: already gone"
        else
            log "  $tbl: $STATUS → deleting"
            try "aws dynamodb delete-table --table-name $tbl --query 'TableDescription.TableStatus' --output text"
        fi
    done
    # Wait loop — DDB deletes take 10-30s each
    log "Waiting for DDB tables to finish deleting..."
    for tbl in medmcqa_questions medmcqa_submissions medmcqa_user_sessions; do
        for i in {1..30}; do
            STATUS=$(aws dynamodb describe-table --table-name "$tbl" \
                --query 'Table.TableStatus' --output text 2>/dev/null || echo "DELETED")
            if [[ "$STATUS" == "DELETED" ]]; then
                log "  $tbl: deleted ✓"
                break
            fi
            sleep 2
        done
    done
else
    log "Skipping DDB deletion (--keep-data)"
fi

# 6b. Cognito user pool
if [[ $KEEP_USERS -eq 0 ]]; then
    log ""
    log "--- Cognito User Pool ---"
    USER_POOL_ID=$(aws cognito-idp list-user-pools --max-results 60 \
        --query "UserPools[?Name=='medmcqa-users'].Id | [0]" --output text 2>/dev/null || echo "None")
    if [[ "$USER_POOL_ID" == "None" || -z "$USER_POOL_ID" ]]; then
        log "  medmcqa-users: already gone"
    else
        log "  medmcqa-users: $USER_POOL_ID → deleting"
        # User pool delete requires no domain attached; detach if present
        DOMAIN=$(aws cognito-idp describe-user-pool --user-pool-id "$USER_POOL_ID" \
            --query 'UserPool.Domain' --output text 2>/dev/null || echo "None")
        if [[ "$DOMAIN" != "None" && -n "$DOMAIN" ]]; then
            log "  Detaching domain: $DOMAIN"
            try "aws cognito-idp delete-user-pool-domain --user-pool-id '$USER_POOL_ID' --domain '$DOMAIN'"
        fi
        try "aws cognito-idp delete-user-pool --user-pool-id '$USER_POOL_ID'"
        # Verify
        VERIFY=$(aws cognito-idp describe-user-pool --user-pool-id "$USER_POOL_ID" \
            --query 'UserPool.Id' --output text 2>/dev/null || echo "MISSING")
        if [[ "$VERIFY" == "MISSING" ]]; then
            log "  User pool deleted ✓"
        else
            warn "User pool may still exist: $VERIFY"
        fi
    fi
else
    log "Skipping Cognito deletion (--keep-users)"
fi

# ── 7. SSM parameters ──────────────────────────────────────────────────────────
log ""
log "--- SSM parameters ---"
# /medmcqa/demo_mode is managed by the stack, but in case the stack failed to
# clean it up, delete it explicitly
try "aws ssm delete-parameter --name /medmcqa/demo_mode --region $AWS_DEFAULT_REGION"

if [[ $PURGE_SSM -eq 1 ]]; then
    log "Purging all /medmcqa/* parameters (--purge-ssm)"
    # List, then delete all under /medmcqa/
    PARAMS=$(aws ssm describe-parameters --region "$AWS_DEFAULT_REGION" \
        --parameter-filters "Key=Name,Option=BeginsWith,Values=/medmcqa/" \
        --query 'Parameters[].Name' --output text 2>/dev/null || echo "")
    if [[ -n "$PARAMS" ]]; then
        for p in $PARAMS; do
            try "aws ssm delete-parameter --name '$p' --region $AWS_DEFAULT_REGION"
        done
    else
        log "  No /medmcqa/* parameters found"
    fi
else
    log "Preserving /medmcqa/hf_token (use --purge-ssm to delete)"
fi

# ── 8. CloudWatch log groups (Lambda auto-created — CDK doesn't always clean) ──
log ""
log "--- CloudWatch log groups ---"
for prefix in /aws/lambda/MedMCQAStack /aws/states/medmcqa-grading; do
    GROUPS=$(aws logs describe-log-groups --log-group-name-prefix "$prefix" \
        --region "$AWS_DEFAULT_REGION" --query 'logGroups[].logGroupName' --output text 2>/dev/null || echo "")
    if [[ -n "$GROUPS" ]]; then
        for lg in $GROUPS; do
            try "aws logs delete-log-group --log-group-name '$lg' --region $AWS_DEFAULT_REGION"
        done
    fi
done

# ── 9. Final verification ──────────────────────────────────────────────────────
log ""
log "=== FINAL VERIFICATION ==="

log "--- Stack ---"
STACK_STATUS=$(aws cloudformation describe-stacks --stack-name MedMCQAStack \
    --region "$AWS_DEFAULT_REGION" --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo "MISSING")
log "  MedMCQAStack: $STACK_STATUS"

log "--- DDB tables ---"
for tbl in medmcqa_questions medmcqa_submissions medmcqa_user_sessions; do
    STATUS=$(aws dynamodb describe-table --table-name "$tbl" \
        --query 'Table.TableStatus' --output text 2>/dev/null || echo "MISSING")
    log "  $tbl: $STATUS"
done

log "--- Cognito user pool ---"
USER_POOL_ID=$(aws cognito-idp list-user-pools --max-results 60 \
    --query "UserPools[?Name=='medmcqa-users'].Id | [0]" --output text 2>/dev/null || echo "None")
log "  medmcqa-users: $USER_POOL_ID"

log "--- DNS probe ---"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "https://dl.mdaie-sutd.fit/" || echo "000")
log "  GET https://dl.mdaie-sutd.fit → HTTP $HTTP_CODE  (000/connect failure expected after teardown)"

log ""
log "=== TEARDOWN COMPLETE ==="
log ""
log "Preserved:"
log "  • CDKToolkit in ap-southeast-1 and us-east-1 (redeploy-ready)"
log "  • Route 53 hosted zone for mdaie-sutd.fit"
if [[ $PURGE_SSM -eq 0 ]]; then
    log "  • /medmcqa/hf_token"
fi
if [[ $KEEP_DATA -eq 1 ]]; then
    log "  • DDB tables (per --keep-data)"
fi
if [[ $KEEP_USERS -eq 1 ]]; then
    log "  • Cognito user pool (per --keep-users)"
fi
log ""
log "To redeploy:  bash aws/cdk_deploy.sh"
log "Full log:     $LOG"
