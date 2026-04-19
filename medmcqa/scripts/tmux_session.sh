#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# SSH into DGX9 and create or attach a named tmux session.
# Supports multi-window layout for parallel monitoring.
#
# Usage:
#   bash scripts/tmux_session.sh                    # attach to session "medmcqa"
#   SESSION=training bash scripts/tmux_session.sh   # custom session name
#   bash scripts/tmux_session.sh --new              # always create fresh session
#
# Tip: To detach from tmux without killing, press Ctrl+B then D.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

DGX_USER="${DGX_USER:-$(whoami)}"
DGX_IP="${DGX_IP:-192.168.0.219}"
SESSION="${SESSION:-medmcqa}"
FORCE_NEW="${1:-}"

if [[ "$FORCE_NEW" == "--new" ]]; then
  # Kill existing session and create fresh with multi-window layout
  ssh -t "$DGX_USER@$DGX_IP" "
    tmux kill-session -t '$SESSION' 2>/dev/null || true
    tmux new-session -d -s '$SESSION' -n 'main' -x 220 -y 50
    tmux new-window   -t '$SESSION' -n 'monitor'
    tmux new-window   -t '$SESSION' -n 'logs'
    tmux send-keys    -t '$SESSION:main' 'cd ~/medmcqa && conda activate medmcqa' Enter
    tmux send-keys    -t '$SESSION:monitor' 'watch -n 2 nvidia-smi' Enter
    tmux select-window -t '$SESSION:main'
    tmux attach-session -t '$SESSION'
  "
else
  # Create if not exists, then attach
  ssh -t "$DGX_USER@$DGX_IP" "
    if tmux has-session -t '$SESSION' 2>/dev/null; then
      echo '→ Attaching to existing tmux session: $SESSION'
      tmux attach-session -t '$SESSION'
    else
      echo '→ Creating new tmux session: $SESSION'
      tmux new-session -d -s '$SESSION' -n 'main' -x 220 -y 50
      tmux new-window  -t '$SESSION' -n 'monitor'
      tmux send-keys   -t '$SESSION:main'    'cd ~/medmcqa && conda activate medmcqa' Enter
      tmux send-keys   -t '$SESSION:monitor' 'watch -n 2 nvidia-smi' Enter
      tmux select-window -t '$SESSION:main'
      tmux attach-session -t '$SESSION'
    fi
  "
fi
