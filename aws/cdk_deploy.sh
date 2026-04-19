#!/usr/bin/env bash
#
# Full CDK deploy of MedMCQAStack to ap-southeast-1.
#
# Steps:
#   1. Load creds from aws/cdk/.env
#   2. Verify AWS reachable + identity
#   3. Set up Python venv with aws-cdk-lib
#   4. Install/use CDK CLI (npx aws-cdk)
#   5. Run `cdk bootstrap` if needed (in BOTH ap-southeast-1 and us-east-1
#      because the cert lives in us-east-1)
#   6. `cdk synth` (catches latent stack errors before deploy)
#   7. `cdk deploy --require-approval never`
#   8. Capture outputs (UserPoolId, ApiEndpoint, CloudFrontUrl, etc.)
#   9. Probe https://dl.mdaie-sutd.fit
#
# All output → aws/cdk_deploy.log
# First-time deploy: 15-25 min (CloudFront + ACM cert validation are the slow parts)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$SCRIPT_DIR/cdk_deploy.log"
: > "$LOG"

log()  { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
fail() { log "FATAL: $*"; log "See full log at $LOG"; exit 1; }
run()  { log "+ $*"; eval "$@" 2>&1 | tee -a "$LOG"; return ${PIPESTATUS[0]}; }

log "=== MedMCQAStack CDK deploy ==="
log "Script dir: $SCRIPT_DIR"
log "Log: $LOG"

# ── 1. Load creds ──────────────────────────────────────────────────────────────
ENV_FILE="$SCRIPT_DIR/cdk/.env"
[[ -f "$ENV_FILE" ]] || fail ".env not found at $ENV_FILE"
log "Sourcing $ENV_FILE"
set -a; source "$ENV_FILE"; set +a
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-ap-southeast-1}"
export CDK_DEFAULT_REGION="${CDK_DEFAULT_REGION:-$AWS_DEFAULT_REGION}"
export CDK_DEFAULT_ACCOUNT="${CDK_DEFAULT_ACCOUNT:-}"
log "Primary region: $AWS_DEFAULT_REGION"
log "CDK region:     $CDK_DEFAULT_REGION"

# ── 2. Toolchain ───────────────────────────────────────────────────────────────
command -v aws    >/dev/null 2>&1 || fail "'aws' CLI missing. brew install awscli"
command -v python3 >/dev/null 2>&1 || fail "'python3' missing"
command -v npx    >/dev/null 2>&1 || fail "'npx' missing. brew install node"

log "aws:    $(aws --version 2>&1)"
log "python: $(python3 --version 2>&1)"
log "node:   $(node --version 2>&1)"

# ── 3. Identity ────────────────────────────────────────────────────────────────
log "--- whoami ---"
if ! aws sts get-caller-identity >>"$LOG" 2>&1; then
    fail "STS get-caller-identity failed — creds invalid"
fi
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
[[ -z "$CDK_DEFAULT_ACCOUNT" ]] && export CDK_DEFAULT_ACCOUNT="$ACCOUNT"
log "Account: $ACCOUNT"

# ── 4. Python venv ─────────────────────────────────────────────────────────────
CDK_DIR="$SCRIPT_DIR/cdk"
cd "$CDK_DIR" || fail "cd $CDK_DIR failed"
log "--- Python venv setup ---"
if [[ ! -d ".venv" ]]; then
    log "Creating .venv"
    run "python3 -m venv .venv" || fail "venv creation failed"
fi
# shellcheck source=/dev/null
source .venv/bin/activate
# Belt-and-suspenders: cdk.json's `app` field already points at .venv/bin/python3,
# but explicitly export VIRTUAL_ENV + prepend bin to PATH so any subshell spawned
# by npx also picks up the venv (npx normally inherits PATH but not activation state).
export VIRTUAL_ENV="$CDK_DIR/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
log "VIRTUAL_ENV: $VIRTUAL_ENV"
log "which python3: $(which python3)"
run "pip install --quiet --upgrade pip" || fail "pip upgrade failed"
run "pip install --quiet -r requirements.txt" || fail "pip install requirements failed"
log "Installed: $(pip show aws-cdk-lib 2>/dev/null | grep -E '^(Name|Version):' | tr '\n' ' ')"

# Sanity-check: the venv's python can actually import aws_cdk
if ! .venv/bin/python3 -c "import aws_cdk; print(f'aws_cdk imported from {aws_cdk.__file__}')" >>"$LOG" 2>&1; then
    fail "venv python3 cannot import aws_cdk — pip install silently failed"
fi

# ── 5. CDK CLI via npx ─────────────────────────────────────────────────────────
log "--- CDK CLI ---"
CDK="npx --yes aws-cdk@2"
log "CDK: $CDK"
run "$CDK --version" || fail "cdk --version failed"

# ── 5b. Preflight orphan cleanup (idempotent) ──────────────────────────────────
# Every previous failed deploy can leave behind any combination of:
#   - A CloudFormation stack in ROLLBACK_COMPLETE (must be deleted before retry)
#   - A Route 53 A-alias record pointing at a now-dead CloudFront target
#   - A CloudFront distribution still claiming dl.mdaie-sutd.fit (409 CNAME conflict)
#   - RETAIN-policy DDB tables / Cognito pool / S3 frontend bucket / SSM demo_mode
# Clean them all up before attempting deploy. All sub-scripts are no-ops when the
# orphan doesn't exist, so this is safe to run on every invocation.
log "--- Preflight: clearing any orphan resources from prior failed deploys ---"

# 5b.1 — If MedMCQAStack is in ROLLBACK_COMPLETE, delete the stack record itself.
STACK_STATUS=$(aws cloudformation describe-stacks --stack-name MedMCQAStack \
                 --region "$AWS_DEFAULT_REGION" \
                 --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo "MISSING")
log "MedMCQAStack status: $STACK_STATUS"
if [[ "$STACK_STATUS" == "ROLLBACK_COMPLETE" || "$STACK_STATUS" == "CREATE_FAILED" ]]; then
    log "  Stack is in $STACK_STATUS — deleting stack record (orphan resources will be handled separately)"
    run "aws cloudformation delete-stack --stack-name MedMCQAStack --region $AWS_DEFAULT_REGION"
    # Wait up to 3 min for the stack record to disappear
    for i in {1..18}; do
        S=$(aws cloudformation describe-stacks --stack-name MedMCQAStack \
              --region "$AWS_DEFAULT_REGION" --query 'Stacks[0].StackStatus' \
              --output text 2>/dev/null || echo "MISSING")
        [[ "$S" == "MISSING" ]] && { log "  Stack record deleted ✓"; break; }
        log "  [$((i*10))s] $S..."
        sleep 10
    done
fi

# 5b.2 — Route 53 orphan (most common, cheapest to clear).
#       NB: use `-f` not `-x` — if the script is missing its execute bit we still
#       invoke it via `bash <path>`; the executable check previously silently
#       skipped this cleanup step (see CLAUDE.md mistakes log #20).
if [[ -f "$SCRIPT_DIR/cleanup_orphan_route53.sh" ]]; then
    run "bash '$SCRIPT_DIR/cleanup_orphan_route53.sh' --yes" || log "  (route53 cleanup returned non-zero, continuing)"
else
    log "  WARN: cleanup_orphan_route53.sh missing — preflight incomplete"
fi

# 5b.3 — CloudFront alias conflict (detach-only fast path)
if [[ -f "$SCRIPT_DIR/cleanup_orphan_cloudfront.sh" ]]; then
    run "bash '$SCRIPT_DIR/cleanup_orphan_cloudfront.sh' --detach-only --yes" || log "  (cloudfront cleanup returned non-zero, continuing)"
else
    log "  WARN: cleanup_orphan_cloudfront.sh missing — preflight incomplete"
fi

# 5b.4 — RETAIN-policy orphans (DDB / Cognito / S3 / SSM)
#       Only runs if tables are empty; --force would be needed for non-empty tables.
if [[ -f "$SCRIPT_DIR/cleanup_retain_orphans.sh" ]]; then
    run "bash '$SCRIPT_DIR/cleanup_retain_orphans.sh' --yes" || log "  (retain cleanup returned non-zero, continuing — may mean DDB has data; pass --force to override)"
else
    log "  WARN: cleanup_retain_orphans.sh missing — preflight incomplete"
fi

# 5b.5 — POST-CLEANUP VERIFICATION. Hard-fail if any known orphan class survived.
#        This catches scripts that exit 0 but silently no-op'd (wrong zone, bad
#        creds, forgotten chmod, etc). Abort the deploy rather than march into
#        another "already exists" rollback.
log "--- Preflight verification (hard-fails if orphans survived cleanup) ---"

SURVIVORS=0

# Route 53 — any record under dl.mdaie-sutd.fit. in zone Z089465439W0RXJJ0LOEW?
R53_REMAIN=$(aws route53 list-resource-record-sets \
    --hosted-zone-id Z089465439W0RXJJ0LOEW \
    --query "ResourceRecordSets[?Name=='dl.mdaie-sutd.fit.'] | length(@)" \
    --output text 2>/dev/null || echo "?")
log "  Route53 records at dl.mdaie-sutd.fit.: $R53_REMAIN"
[[ "$R53_REMAIN" != "0" ]] && SURVIVORS=$((SURVIVORS + 1))

# CloudFront — any distribution still claiming the alias?
CF_REMAIN=$(aws cloudfront list-distributions \
    --query "DistributionList.Items[?Aliases.Items[?@=='dl.mdaie-sutd.fit']] | length(@)" \
    --output text 2>/dev/null || echo "?")
log "  CloudFront distributions with alias dl.mdaie-sutd.fit: $CF_REMAIN"
[[ "$CF_REMAIN" != "0" ]] && SURVIVORS=$((SURVIVORS + 1))

# DDB — any of the three tables still present?
for tbl in medmcqa_questions medmcqa_submissions medmcqa_user_sessions; do
    S=$(aws dynamodb describe-table --table-name "$tbl" \
        --query 'Table.TableStatus' --output text 2>/dev/null || echo "MISSING")
    log "  DDB $tbl: $S"
    [[ "$S" != "MISSING" ]] && SURVIVORS=$((SURVIVORS + 1))
done

# Cognito — any pool named medmcqa-users?
CP=$(aws cognito-idp list-user-pools --max-results 60 \
    --query "UserPools[?Name=='medmcqa-users'] | length(@)" --output text 2>/dev/null || echo "?")
log "  Cognito pools named medmcqa-users: $CP"
[[ "$CP" != "0" ]] && SURVIVORS=$((SURVIVORS + 1))

if [[ "$SURVIVORS" -gt 0 ]]; then
    log ""
    log "ERROR: $SURVIVORS orphan(s) survived preflight cleanup. Refusing to run cdk deploy — it would fail."
    log ""
    log "Manual next step — run the cleanup scripts individually and see where they fail:"
    log "  bash $SCRIPT_DIR/cleanup_orphan_route53.sh --dry-run"
    log "  bash $SCRIPT_DIR/cleanup_orphan_cloudfront.sh --dry-run"
    log "  bash $SCRIPT_DIR/cleanup_retain_orphans.sh   # without --yes to see inventory"
    log ""
    log "Common causes: missing chmod +x, wrong hosted zone ID, expired AWS creds, different region."
    fail "Preflight verification failed — $SURVIVORS orphan resource class(es) remain"
fi

log "Preflight cleanup + verification complete — environment is pristine."
log ""

# ── 6. Bootstrap (idempotent — checks first) ───────────────────────────────────
bootstrap_if_needed() {
    local region="$1"
    local cf_status
    cf_status=$(aws cloudformation describe-stacks --stack-name CDKToolkit \
                  --region "$region" --query 'Stacks[0].StackStatus' --output text 2>/dev/null \
                  || echo "MISSING")
    if [[ "$cf_status" == "MISSING" ]]; then
        log "Bootstrapping CDK in $region (one-time setup, ~2 min)"
        run "$CDK bootstrap aws://$ACCOUNT/$region" \
            || fail "cdk bootstrap $region failed"
    else
        log "Bootstrap exists in $region (CDKToolkit: $cf_status)"
    fi
}
bootstrap_if_needed "$AWS_DEFAULT_REGION"
bootstrap_if_needed "us-east-1"   # required because ACM cert is in us-east-1

# ── 7. Synth (validates the stack) ─────────────────────────────────────────────
log "--- cdk synth ---"
if ! run "$CDK synth --quiet"; then
    fail "cdk synth failed — fix stack errors before deploying"
fi

# ── 8. Diff (informational) ────────────────────────────────────────────────────
log "--- cdk diff (informational; non-fatal) ---"
$CDK diff 2>&1 | tee -a "$LOG" || true

# ── 9. Deploy ──────────────────────────────────────────────────────────────────
log "--- cdk deploy (this can take 15-25 min on first run) ---"
log "  CloudFront distributions: ~10-15 min to propagate"
log "  ACM cert DNS validation:  ~5 min"
DEPLOY_START=$(date +%s)
if ! run "$CDK deploy --require-approval never --outputs-file cdk-outputs.json"; then
    fail "cdk deploy failed — see log for the exact CloudFormation event that errored"
fi
DEPLOY_END=$(date +%s)
log "Deploy time: $(( (DEPLOY_END - DEPLOY_START) / 60 )) min $(( (DEPLOY_END - DEPLOY_START) % 60 )) sec"

# ── 10. Outputs ────────────────────────────────────────────────────────────────
log ""
log "=== STACK OUTPUTS ==="
if [[ -f cdk-outputs.json ]]; then
    cat cdk-outputs.json | python3 -m json.tool | tee -a "$LOG"
fi

# ── 10.5. Frontend deploy ──────────────────────────────────────────────────────
# CDK only creates the empty S3 bucket + CloudFront. We have to build+sync the
# React bundle ourselves. Without this step, https://dl.mdaie-sutd.fit returns
# a 403 AccessDenied from S3 (CloudFront faithfully forwards to an empty bucket).
# The sub-script also rewrites .env.production from cdk-outputs.json, so pool /
# client / API-endpoint IDs can't drift between infra and the JS bundle.
if [[ -f "$SCRIPT_DIR/deploy_frontend.sh" ]]; then
    log ""
    log "--- Frontend build + S3 sync + CF invalidation ---"
    run "bash '$SCRIPT_DIR/deploy_frontend.sh'" || fail "frontend deploy failed"
else
    log "WARN: deploy_frontend.sh missing — site will 403 until frontend is uploaded"
fi

# ── 11. Verification probes ────────────────────────────────────────────────────
log ""
log "=== VERIFICATION ==="

log "--- DDB tables ---"
for tbl in medmcqa_questions medmcqa_submissions medmcqa_user_sessions; do
    STATUS=$(aws dynamodb describe-table --table-name "$tbl" \
        --query 'Table.TableStatus' --output text 2>/dev/null || echo "MISSING")
    log "  $tbl: $STATUS"
done

log "--- Lambda functions ---"
aws lambda list-functions \
    --query "Functions[?starts_with(FunctionName, 'MedMCQAStack')].{Name:FunctionName,Runtime:Runtime,Modified:LastModified}" \
    --output table 2>>"$LOG" | tee -a "$LOG"

log "--- Step Functions ---"
aws stepfunctions list-state-machines \
    --query "stateMachines[?name=='medmcqa-grading']" \
    --output table 2>>"$LOG" | tee -a "$LOG"

log "--- Cognito User Pool ---"
USER_POOL_ID=$(python3 -c "import json; d=json.load(open('cdk-outputs.json'))['MedMCQAStack']; print(d.get('UserPoolId',''))" 2>/dev/null || echo "")
log "  UserPoolId: $USER_POOL_ID"

log "--- Live URL probe ---"
URL="https://dl.mdaie-sutd.fit"
log "  Waiting 30s for DNS + CloudFront warm-up..."
sleep 30
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "$URL/" || echo "000")
log "  GET $URL → HTTP $HTTP_CODE   (200 = SPA loaded, 403 = empty bucket, 000 = DNS not propagated yet)"
API_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 "$URL/api/questions" || echo "000")
log "  GET $URL/api/questions → HTTP $API_CODE   (401 expected = JWT auth working)"

log ""
log "=== CDK DEPLOY COMPLETE ==="
log "Next: bash aws/deploy_demo.sh   (pushes the DEMO_MODE patch to the now-existing GradeFn)"
log "Full log: $LOG"
