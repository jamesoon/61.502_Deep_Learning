#!/usr/bin/env bash
#
# Deploy DEMO_MODE changes to the live MedMCQAStack on ap-southeast-1.
#
# What this does (Path B — Lambda-only, no CloudFormation churn):
#   1. Loads AWS creds from aws/cdk/.env
#   2. Creates/updates SSM parameter /medmcqa/demo_mode (default value "0")
#   3. Repackages the grade Lambda with handler.py + demo_explanations.json
#   4. Pushes the new code to MedMCQAStack-GradeFn-*
#   5. Adds DEMO_MODE=0 env var as the env-fallback (SSM is authoritative)
#   6. Verifies — re-reads SSM, env, and the deployed function metadata
#   7. Verifies the live URL https://dl.mdaie-sutd.fit responds
#
# All output is tee'd to deploy.log next to this script.
# Exit code 0 = full success; non-zero = something failed (check the log).
#
# Run from the repo root OR from the aws/ folder; both work.

set -uo pipefail

# ── Locate ourselves ───────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$SCRIPT_DIR/deploy.log"
: > "$LOG"   # truncate

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
fail() { log "FATAL: $*"; exit 1; }
run()  { log "+ $*"; eval "$@" 2>&1 | tee -a "$LOG"; return ${PIPESTATUS[0]}; }

log "=== MedMCQA demo-mode deploy ==="
log "Script dir: $SCRIPT_DIR"
log "Log: $LOG"

# ── 1. Load creds ──────────────────────────────────────────────────────────────
ENV_FILE="$SCRIPT_DIR/cdk/.env"
[[ -f "$ENV_FILE" ]] || fail ".env not found at $ENV_FILE"
log "Sourcing $ENV_FILE"
set -a; source "$ENV_FILE"; set +a
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-ap-southeast-1}"
log "Region: $AWS_DEFAULT_REGION"

# ── 2. Toolchain checks ────────────────────────────────────────────────────────
command -v aws >/dev/null 2>&1 || fail "'aws' CLI not on PATH. Install: brew install awscli"
command -v zip >/dev/null 2>&1 || fail "'zip' not on PATH"
log "aws version: $(aws --version 2>&1)"

# ── 3. Identity check ──────────────────────────────────────────────────────────
log "--- whoami ---"
if ! aws sts get-caller-identity >>"$LOG" 2>&1; then
    fail "STS get-caller-identity failed — creds in .env are likely expired or wrong"
fi
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
log "Account: $ACCOUNT"

# ── 4. Locate the live grade Lambda ────────────────────────────────────────────
log "--- locating GradeFn ---"
GRADE_FN=$(aws lambda list-functions \
  --query "Functions[?starts_with(FunctionName, 'MedMCQAStack-GradeFn')].FunctionName | [0]" \
  --output text 2>>"$LOG")
[[ "$GRADE_FN" != "None" && -n "$GRADE_FN" ]] || fail "MedMCQAStack-GradeFn-* not found. Is the stack deployed in $AWS_DEFAULT_REGION?"
log "Grade function: $GRADE_FN"

# ── 5. SSM parameter ───────────────────────────────────────────────────────────
log "--- SSM /medmcqa/demo_mode ---"
EXISTING_VALUE=$(aws ssm get-parameter --name /medmcqa/demo_mode \
                  --query 'Parameter.Value' --output text 2>/dev/null || echo "MISSING")
if [[ "$EXISTING_VALUE" == "MISSING" ]]; then
    log "Parameter does not exist — creating with value '0'"
    run "aws ssm put-parameter --name /medmcqa/demo_mode --type String --value 0" \
        || fail "ssm put-parameter failed"
else
    log "Parameter already exists (value=$EXISTING_VALUE) — leaving as-is"
fi

# ── 6. Package & push grade Lambda code ────────────────────────────────────────
log "--- packaging grade Lambda ---"
GRADE_DIR="$SCRIPT_DIR/lambdas/grade"
[[ -f "$GRADE_DIR/handler.py" ]] || fail "handler.py not found in $GRADE_DIR"
[[ -f "$GRADE_DIR/demo_explanations.json" ]] || fail "demo_explanations.json not found in $GRADE_DIR"

ZIP_FILE="/tmp/grade-$$-$(date +%s).zip"
( cd "$GRADE_DIR" && zip -r "$ZIP_FILE" handler.py demo_explanations.json >>"$LOG" 2>&1 ) \
    || fail "zip failed"
log "Wrote $ZIP_FILE ($(du -h "$ZIP_FILE" | cut -f1))"

log "--- update-function-code ---"
run "aws lambda update-function-code --function-name '$GRADE_FN' --zip-file 'fileb://$ZIP_FILE' --no-cli-pager" \
    || fail "update-function-code failed"

# Wait for code update to settle before updating config (AWS race)
log "Waiting for function update to settle..."
aws lambda wait function-updated --function-name "$GRADE_FN" >>"$LOG" 2>&1 \
    || fail "function-updated wait failed"

# ── 7. Merge DEMO_MODE=0 into env (preserves existing vars) ─────────────────────
log "--- merging DEMO_MODE env var ---"
EXISTING_ENV_JSON=$(aws lambda get-function-configuration \
    --function-name "$GRADE_FN" \
    --query 'Environment.Variables' --output json)

NEW_ENV_JSON=$(echo "$EXISTING_ENV_JSON" | python3 -c "
import json, sys
e = json.load(sys.stdin) or {}
e['DEMO_MODE'] = '0'
print(json.dumps({'Variables': e}))
")
log "New env (keys): $(echo "$NEW_ENV_JSON" | python3 -c 'import json,sys; print(sorted(json.load(sys.stdin)["Variables"].keys()))')"

run "aws lambda update-function-configuration --function-name '$GRADE_FN' --environment '$NEW_ENV_JSON' --no-cli-pager" \
    || fail "update-function-configuration failed"
aws lambda wait function-updated --function-name "$GRADE_FN" >>"$LOG" 2>&1

rm -f "$ZIP_FILE"

# ── 8. Verification ────────────────────────────────────────────────────────────
log ""
log "=== VERIFICATION ==="

log "--- SSM read-back ---"
aws ssm get-parameter --name /medmcqa/demo_mode --output table | tee -a "$LOG"

log "--- Lambda env read-back ---"
aws lambda get-function-configuration --function-name "$GRADE_FN" \
    --query '{Runtime:Runtime,Updated:LastModified,CodeSha:CodeSha256,DemoMode:Environment.Variables.DEMO_MODE,HFExplain:Environment.Variables.HF_EXPLAIN_MODEL}' \
    --output table | tee -a "$LOG"

log "--- Step Functions state machine ---"
SM_ARN=$(aws stepfunctions list-state-machines \
    --query "stateMachines[?name=='medmcqa-grading'].stateMachineArn | [0]" \
    --output text 2>/dev/null || echo "")
if [[ -n "$SM_ARN" && "$SM_ARN" != "None" ]]; then
    log "State machine: $SM_ARN"
else
    log "WARN: state machine 'medmcqa-grading' not found"
fi

log "--- DynamoDB tables ---"
for tbl in medmcqa_questions medmcqa_submissions medmcqa_user_sessions; do
    STATUS=$(aws dynamodb describe-table --table-name "$tbl" \
        --query 'Table.TableStatus' --output text 2>/dev/null || echo "MISSING")
    log "  $tbl: $STATUS"
done

log "--- Live URL probe ---"
URL="https://dl.mdaie-sutd.fit"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$URL/" || echo "000")
log "  GET $URL → HTTP $HTTP_CODE  (200 = SPA loaded; 403 = bucket misconfig)"
API_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$URL/api/questions" || echo "000")
log "  GET $URL/api/questions → HTTP $API_CODE  (401 expected — auth required)"

log ""
log "=== DEPLOY COMPLETE ==="
log "To enable demo mode for showtime:"
log "  aws ssm put-parameter --name /medmcqa/demo_mode --value 1 --overwrite --type String"
log "  aws lambda update-function-configuration --function-name '$GRADE_FN' --description \"demo mode on \$(date -u +%FT%TZ)\""
log ""
log "Full log: $LOG"
