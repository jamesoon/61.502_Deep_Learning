#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Run this on your LOCAL machine to check DGX9 training status without
# needing to attach to tmux. Prints a quick one-shot status snapshot.
#
# Usage (from local machine, in the medmcqa directory):
#   bash scripts/remote_status.sh
#   watch -n 30 bash scripts/remote_status.sh   # auto-refresh every 30s
# ─────────────────────────────────────────────────────────────────────────────

DGX_USER="${DGX_USER:-$(whoami)}"
DGX_IP="${DGX_IP:-192.168.0.219}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  DGX9 Training Status — $(date '+%Y-%m-%d %H:%M:%S')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

ssh -o ConnectTimeout=5 "$DGX_USER@$DGX_IP" bash <<'REMOTE'
  cd ~/medmcqa 2>/dev/null || { echo "  ~/medmcqa not found on DGX9."; exit 1; }

  echo ""
  echo "▶ GPU"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader,nounits 2>/dev/null | \
  awk -F', ' '{printf "  %-20s  %d/%d MB  GPU:%d%%  %d°C\n", $1,$2,$3,$4,$5}'

  echo ""
  echo "▶ Training Processes"
  pgrep -af "python.*train\|torchrun" 2>/dev/null | cut -c1-80 | sed 's/^/  /' \
    || echo "  None running."

  echo ""
  echo "▶ Latest Metrics"
  for ckpt_dir in checkpoints/lora checkpoints/sft; do
    STATE=$(find "$ckpt_dir" -name "trainer_state.json" 2>/dev/null | sort | tail -1)
    if [[ -n "$STATE" ]]; then
      label=$(echo "$ckpt_dir" | sed 's|checkpoints/||')
      printf "  [%s] " "$label"
      python3 - "$STATE" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    s = json.load(f)
logs = [l for l in s.get("log_history",[]) if "loss" in l and "eval_loss" not in l]
evals = [l for l in s.get("log_history",[]) if "eval_loss" in l]
if logs:
    l = logs[-1]
    loss_str = f"{l['loss']:.4f}" if isinstance(l.get('loss'), float) else str(l.get('loss','?'))
    print(f"step {l.get('step','?')}/{s.get('max_steps','?')}  "
          f"epoch {l.get('epoch','?')}/{s.get('num_train_epochs','?')}  "
          f"train_loss={loss_str}", end="")
if evals:
    print(f"  eval_loss={evals[-1].get('eval_loss',0):.4f}", end="")
print()
PYEOF
    fi
  done

  echo ""
  echo "▶ Disk"
  du -sh checkpoints/ results/ 2>/dev/null | sed 's/^/  /' || echo "  (empty)"
REMOTE
