#!/usr/bin/env bash
#
# Cleanup orphan Route 53 records left behind by a failed cdk deploy.
#
# WHY THIS EXISTS:
#   When cdk deploy fails partway through and the rollback can't delete an
#   alias record (typically because its CloudFront target was already gone),
#   the record is left behind. The next cdk deploy then fails with:
#     "Tried to create resource record set [name='dl.mdaie-sutd.fit.', type='A']
#      but it already exists"
#
# WHAT THIS SCRIPT DOES:
#   - Lists records in hosted zone Z089465439W0RXJJ0LOEW (mdaie-sutd.fit)
#     matching the project subdomain (dl.mdaie-sutd.fit)
#   - For each match, builds a ChangeBatch{Action=DELETE} and submits it
#   - Verifies the records are gone
#
# It does NOT touch:
#   - The hosted zone itself
#   - Any record that doesn't match the project subdomain
#
# Flags:
#   --yes      Skip confirmation
#   --dry-run  Show what would be deleted, don't change anything
#
# Usage:
#   bash aws/cleanup_orphan_route53.sh --dry-run
#   bash aws/cleanup_orphan_route53.sh --yes

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$SCRIPT_DIR/route53_cleanup.log"
: > "$LOG"

HOSTED_ZONE_ID="Z089465439W0RXJJ0LOEW"
SUBDOMAIN_FQDN="dl.mdaie-sutd.fit."   # trailing dot required by Route 53

YES=0
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --yes|-y)      YES=1 ;;
        --dry-run|-n)  DRY_RUN=1 ;;
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

# Load creds
ENV_FILE="$SCRIPT_DIR/cdk/.env"
[[ -f "$ENV_FILE" ]] || fail ".env not found at $ENV_FILE"
set -a; source "$ENV_FILE"; set +a

aws sts get-caller-identity >>"$LOG" 2>&1 || fail "STS get-caller-identity failed"

log "=== Route 53 orphan-record cleanup ==="
log "Hosted zone: $HOSTED_ZONE_ID"
log "Target FQDN: $SUBDOMAIN_FQDN"
log "Dry-run: $DRY_RUN"

command -v jq >/dev/null 2>&1 || fail "jq is required (brew install jq)"

# ── 1. List matching records ───────────────────────────────────────────────────
log ""
log "--- Listing records ---"

TMP=$(mktemp -d)
aws route53 list-resource-record-sets \
    --hosted-zone-id "$HOSTED_ZONE_ID" \
    --query "ResourceRecordSets[?Name=='$SUBDOMAIN_FQDN']" \
    > "$TMP/records.json" 2>>"$LOG" \
    || fail "Failed to list records (creds or zone ID wrong?)"

COUNT=$(jq 'length' "$TMP/records.json")
log "Matching records: $COUNT"

if [[ "$COUNT" -eq 0 ]]; then
    log "No orphan records under $SUBDOMAIN_FQDN — nothing to do."
    log "If cdk deploy still fails with 'already exists', run with -- $SUBDOMAIN_FQDN"
    log "to verify the FQDN matches what CDK is creating."
    exit 0
fi

jq '.' "$TMP/records.json" | tee -a "$LOG"

# ── 2. Confirm ─────────────────────────────────────────────────────────────────
if [[ $DRY_RUN -eq 1 ]]; then
    log ""
    log "[DRY-RUN] Would DELETE the $COUNT records above."
    rm -rf "$TMP"
    exit 0
fi

if [[ $YES -eq 0 ]]; then
    echo ""
    read -r -p "Delete these $COUNT records? Type 'delete' to confirm: " confirm
    [[ "$confirm" == "delete" ]] || { log "Aborted by user."; rm -rf "$TMP"; exit 0; }
fi

# ── 3. Build ChangeBatch and submit ────────────────────────────────────────────
log ""
log "--- Building DELETE ChangeBatch ---"
jq '{
  Comment: "orphan cleanup from failed cdk deploy",
  Changes: map({Action: "DELETE", ResourceRecordSet: .})
}' "$TMP/records.json" > "$TMP/change_batch.json"

cat "$TMP/change_batch.json" | tee -a "$LOG"

log ""
log "--- Submitting change ---"
CHANGE_INFO=$(aws route53 change-resource-record-sets \
    --hosted-zone-id "$HOSTED_ZONE_ID" \
    --change-batch "file://$TMP/change_batch.json" \
    --output json 2>&1 | tee -a "$LOG")
CHANGE_ID=$(echo "$CHANGE_INFO" | jq -r '.ChangeInfo.Id // empty' 2>/dev/null || echo "")

if [[ -z "$CHANGE_ID" ]]; then
    fail "Change submission failed — see log"
fi
log "Change submitted: $CHANGE_ID"

# ── 4. Wait for INSYNC ─────────────────────────────────────────────────────────
log "Waiting for change to propagate (typically 30-60s)..."
for ((i=1; i<=18; i++)); do
    STATUS=$(aws route53 get-change --id "$CHANGE_ID" --query 'ChangeInfo.Status' --output text 2>/dev/null || echo "Unknown")
    if [[ "$STATUS" == "INSYNC" ]]; then
        log "Change $CHANGE_ID: INSYNC ✓"
        break
    fi
    log "  [$((i*5))s] $STATUS..."
    sleep 5
done

# ── 5. Verify ──────────────────────────────────────────────────────────────────
log ""
log "--- Verification ---"
REMAINING=$(aws route53 list-resource-record-sets \
    --hosted-zone-id "$HOSTED_ZONE_ID" \
    --query "ResourceRecordSets[?Name=='$SUBDOMAIN_FQDN'] | length(@)" \
    --output text 2>/dev/null || echo "?")
log "Records under $SUBDOMAIN_FQDN remaining: $REMAINING"

if [[ "$REMAINING" == "0" ]]; then
    log "Cleanup successful ✓"
else
    warn "$REMAINING records still present — check console"
fi

rm -rf "$TMP"
log ""
log "Next: bash aws/cdk_deploy.sh"
log "Full log: $LOG"
