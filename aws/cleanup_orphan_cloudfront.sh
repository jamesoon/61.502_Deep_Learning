#!/usr/bin/env bash
#
# Cleanup orphan CloudFront distributions that own dl.mdaie-sutd.fit.
#
# WHY THIS EXISTS:
#   When `cdk deploy` half-rolls and then rolls back, the CloudFront distribution
#   is sometimes left behind because it takes 10-20 min to delete (it has to
#   propagate "disabled" status to all edge POPs first). Subsequent cdk deploys
#   then fail with:
#     "AWS::CloudFront::Distribution: One or more of the CNAMEs you provided
#      are already associated with a different resource."
#
# WHAT THIS SCRIPT DOES:
#   1. Lists ALL CloudFront distributions in the account (CloudFront is global)
#   2. Finds the ones that have dl.mdaie-sutd.fit in their Aliases
#   3. For each such distribution:
#       a. If it's enabled → disables it (update-distribution Enabled=false)
#          Then waits for status=Deployed (this takes 10-15 min)
#       b. Once disabled and Deployed → deletes it
#   4. After all are gone, the next `cdk deploy` will succeed
#
# Alternative (faster but riskier): the script can also just remove the alias
# from the distribution config without disabling/deleting it (the distribution
# stays alive as an orphan, but no longer blocks the CDK deploy). Use --detach-only.
#
# Usage:
#   bash aws/cleanup_orphan_cloudfront.sh                # interactive, full delete
#   bash aws/cleanup_orphan_cloudfront.sh --yes          # non-interactive
#   bash aws/cleanup_orphan_cloudfront.sh --detach-only  # just remove the alias (fast, ~3 min)
#   bash aws/cleanup_orphan_cloudfront.sh --dry-run      # show what would happen
#
# Output → aws/cloudfront_cleanup.log

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$SCRIPT_DIR/cloudfront_cleanup.log"
: > "$LOG"

ALIAS="dl.mdaie-sutd.fit"

YES=0
DETACH_ONLY=0
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --yes|-y)       YES=1 ;;
        --detach-only)  DETACH_ONLY=1 ;;
        --dry-run|-n)   DRY_RUN=1 ;;
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

log "=== CloudFront orphan-distribution cleanup ==="
log "Target alias: $ALIAS"
log "Mode: $([ $DETACH_ONLY -eq 1 ] && echo 'detach-only (fast)' || echo 'full delete (slow)')"
log "Dry-run: $DRY_RUN"

# Load credentials
ENV_FILE="$SCRIPT_DIR/cdk/.env"
[[ -f "$ENV_FILE" ]] || fail ".env not found at $ENV_FILE"
set -a; source "$ENV_FILE"; set +a

# CloudFront is global — region doesn't matter for these calls

aws sts get-caller-identity >>"$LOG" 2>&1 || fail "STS get-caller-identity failed"
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
log "Account: $ACCOUNT"

# ── 1. Find orphan distributions ───────────────────────────────────────────────
log ""
log "--- Searching CloudFront for distributions claiming '$ALIAS' ---"
ORPHAN_IDS=$(aws cloudfront list-distributions \
    --query "DistributionList.Items[?Aliases.Items[?@=='$ALIAS']].Id" \
    --output text 2>>"$LOG" || echo "")

if [[ -z "$ORPHAN_IDS" ]]; then
    log "No CloudFront distributions found with '$ALIAS' as alias."
    log "If cdk deploy still fails with the same error, the conflict may be in"
    log "Route 53 (alias record pointing at a stale CF domain). Check with:"
    log "  aws route53 list-resource-record-sets --hosted-zone-id Z089465439W0RXJJ0LOEW"
    exit 0
fi

log "Found orphan distribution(s):"
for did in $ORPHAN_IDS; do
    DESCR=$(aws cloudfront get-distribution --id "$did" \
        --query 'Distribution.{Id:Id,Domain:DomainName,Status:Status,Enabled:DistributionConfig.Enabled,Aliases:DistributionConfig.Aliases.Items}' \
        --output json 2>/dev/null)
    log "  $DESCR"
done

# ── 2. Confirm ──────────────────────────────────────────────────────────────────
if [[ $DRY_RUN -eq 1 ]]; then
    log ""
    log "[DRY-RUN] Would process distributions: $ORPHAN_IDS"
    log "[DRY-RUN] Mode: $([ $DETACH_ONLY -eq 1 ] && echo 'detach alias only' || echo 'disable then delete')"
    exit 0
fi

if [[ $YES -eq 0 ]]; then
    echo ""
    if [[ $DETACH_ONLY -eq 1 ]]; then
        read -r -p "Detach '$ALIAS' from these distributions? Type 'detach' to confirm: " confirm
        [[ "$confirm" == "detach" ]] || { log "Aborted."; exit 0; }
    else
        read -r -p "Disable + delete these distributions? Type 'delete' to confirm: " confirm
        [[ "$confirm" == "delete" ]] || { log "Aborted."; exit 0; }
    fi
fi

# ── 3. Process each ────────────────────────────────────────────────────────────
for did in $ORPHAN_IDS; do
    log ""
    log "=== Processing distribution $did ==="

    # Fetch current config + ETag (needed for any update)
    TMP=$(mktemp -d)
    aws cloudfront get-distribution-config --id "$did" > "$TMP/dist.json" 2>>"$LOG" \
        || { warn "Failed to get config for $did — skipping"; continue; }
    ETAG=$(jq -r '.ETag' "$TMP/dist.json")
    jq '.DistributionConfig' "$TMP/dist.json" > "$TMP/config.json"
    log "Got config (ETag: $ETAG)"

    if [[ $DETACH_ONLY -eq 1 ]]; then
        # ── 3a. Detach-only path: strip the alias and update ──────────────────
        log "Removing '$ALIAS' from Aliases.Items"
        jq --arg alias "$ALIAS" '
          .Aliases.Items |= map(select(. != $alias))
          | .Aliases.Quantity = (.Aliases.Items | length)
        ' "$TMP/config.json" > "$TMP/config_new.json"

        # CloudFront requires at least one cert source. If we removed the only alias,
        # we must also strip the custom cert (revert to default cf cert).
        REMAINING=$(jq -r '.Aliases.Quantity' "$TMP/config_new.json")
        if [[ "$REMAINING" == "0" ]]; then
            log "  No aliases remain — reverting to default CloudFront cert"
            jq '
              .ViewerCertificate = {
                "CloudFrontDefaultCertificate": true,
                "MinimumProtocolVersion": "TLSv1",
                "CertificateSource": "cloudfront"
              }
            ' "$TMP/config_new.json" > "$TMP/config_final.json"
            mv "$TMP/config_final.json" "$TMP/config_new.json"
        fi

        log "Calling update-distribution..."
        aws cloudfront update-distribution \
            --id "$did" \
            --if-match "$ETAG" \
            --distribution-config "file://$TMP/config_new.json" \
            --query 'Distribution.{Id:Id,Status:Status,Aliases:DistributionConfig.Aliases.Items}' \
            --output json 2>&1 | tee -a "$LOG" \
            || { warn "Update failed for $did"; continue; }
        log "Detached $ALIAS from $did. The distribution itself remains (orphan)."
        log "You can now re-run cdk_deploy.sh; the new stack will claim the alias."
        rm -rf "$TMP"
        continue
    fi

    # ── 3b. Full delete path: disable, wait, delete ───────────────────────────
    ENABLED=$(jq -r '.Enabled' "$TMP/config.json")
    if [[ "$ENABLED" == "true" ]]; then
        log "Distribution is enabled — disabling first"
        jq '.Enabled = false' "$TMP/config.json" > "$TMP/config_disabled.json"
        aws cloudfront update-distribution \
            --id "$did" \
            --if-match "$ETAG" \
            --distribution-config "file://$TMP/config_disabled.json" \
            --query 'Distribution.Status' \
            --output text 2>&1 | tee -a "$LOG" \
            || { warn "Disable failed for $did"; continue; }
        log "Disable request sent. Waiting for status=Deployed (this takes 10-15 min)..."
        # Poll up to 30 min
        for ((i=1; i<=180; i++)); do
            STATUS=$(aws cloudfront get-distribution --id "$did" \
                --query 'Distribution.Status' --output text 2>/dev/null || echo "Unknown")
            if [[ "$STATUS" == "Deployed" ]]; then
                log "Distribution $did is now Deployed (and disabled)."
                break
            fi
            if (( i % 6 == 0 )); then
                log "  [$((i*10))s] still $STATUS, waiting..."
            fi
            sleep 10
        done
        STATUS=$(aws cloudfront get-distribution --id "$did" \
            --query 'Distribution.Status' --output text 2>/dev/null || echo "Unknown")
        if [[ "$STATUS" != "Deployed" ]]; then
            warn "Timed out waiting for $did to reach Deployed (still $STATUS)."
            warn "Re-run this script to retry the delete once it propagates."
            rm -rf "$TMP"
            continue
        fi
    else
        log "Distribution is already disabled."
    fi

    # Re-fetch ETag (it changes on every update)
    NEW_ETAG=$(aws cloudfront get-distribution --id "$did" \
        --query 'ETag' --output text 2>/dev/null)
    log "Deleting distribution $did (ETag: $NEW_ETAG)"
    aws cloudfront delete-distribution --id "$did" --if-match "$NEW_ETAG" 2>&1 | tee -a "$LOG" \
        || warn "Delete failed for $did (it may already be gone)"

    # Verify
    sleep 3
    EXISTS=$(aws cloudfront get-distribution --id "$did" \
        --query 'Distribution.Id' --output text 2>/dev/null || echo "MISSING")
    if [[ "$EXISTS" == "MISSING" ]]; then
        log "Distribution $did: deleted ✓"
    else
        warn "Distribution $did still exists (status: $(aws cloudfront get-distribution --id "$did" --query 'Distribution.Status' --output text 2>/dev/null))"
    fi

    rm -rf "$TMP"
done

log ""
log "=== CLEANUP COMPLETE ==="
log "Next step:  bash aws/cdk_deploy.sh"
log "Full log:   $LOG"
