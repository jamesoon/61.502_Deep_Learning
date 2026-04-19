"""
_switch_and_run_qwen35.py

Waits for qwen3_8b CoT evaluation to finish, then:
  1. Unloads qwen3_8b from LM Studio via REST API
  2. Loads Qwen3.5-9B-Q4_K_M in LM Studio
  3. Waits for the model to be ready
  4. Runs run_evaluation_cot.py (which is already set to qwen3_5_9b)
"""

import os
import sys
import time
import subprocess
import requests

# =============================================================================
# CONFIG
# =============================================================================
LM_STUDIO_BASE = os.getenv("LOCAL_API_BASE_URL", "http://127.0.0.1:1234")
API_BASE        = f"{LM_STUDIO_BASE}/api/v0"
OPENAI_BASE     = f"{LM_STUDIO_BASE}/v1"

# The HuggingFace repo/file for the model to LOAD
LOAD_MODEL_PATH = "Qwen/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

# Tokens used to detect qwen3_8b is currently loaded (to unload it)
UNLOAD_MATCH_TOKENS = ["qwen3", "8b"]

# Tokens used to confirm qwen3.5-9b is loaded and ready
READY_MATCH_TOKENS  = ["qwen3", "5", "9b"]

# Per-model result dir and the file we watch for qwen3_8b completion
RESULT_DIR   = "results_local_llm_cot"
PER_MODEL_DIR = os.path.join(RESULT_DIR, "per_model")
WATCH_FILE    = os.path.join(PER_MODEL_DIR, "test_set_07__qwen3_8b.csv")
MIN_ROWS_DONE = 225       # consider done when CSV has at least this many data rows

POLL_INTERVAL   = 30      # seconds between checks
LOAD_TIMEOUT    = 300     # seconds to wait for model to load
EVAL_SCRIPT     = "run_evaluation_cot.py"
PYTHON          = os.path.join(".venv", "Scripts", "python.exe")


# =============================================================================
# HELPERS
# =============================================================================
def normalize(text: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def count_csv_rows(path: str) -> int:
    """Count data rows (excluding header) in a CSV file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f) - 1   # subtract header
    except Exception:
        return 0


def list_loaded_models():
    try:
        r = requests.get(f"{API_BASE}/models", timeout=10)
        r.raise_for_status()
        data = r.json()
        # LM Studio returns {"data": [...]} or a list directly
        items = data.get("data", data) if isinstance(data, dict) else data
        return items
    except Exception as e:
        print(f"  [API] Could not list models: {e}")
        return []


def find_model_id(match_tokens):
    """Return the first loaded model whose ID matches all tokens."""
    models = list_loaded_models()
    for m in models:
        mid = m.get("id", "") or m.get("identifier", "")
        if all(t in normalize(mid) for t in match_tokens):
            return mid
    return None


def unload_model(identifier: str) -> bool:
    print(f"  [API] Unloading model: {identifier}")
    try:
        r = requests.post(
            f"{API_BASE}/models/unload",
            json={"identifier": identifier},
            timeout=30,
        )
        print(f"  [API] Unload response: {r.status_code} {r.text[:200]}")
        return r.status_code < 300
    except Exception as e:
        print(f"  [API] Unload error: {e}")
        return False


def load_model(model_path: str) -> bool:
    print(f"  [API] Loading model: {model_path}")
    try:
        r = requests.post(
            f"{API_BASE}/models/load",
            json={"model": model_path},
            timeout=30,
        )
        print(f"  [API] Load response: {r.status_code} {r.text[:200]}")
        return r.status_code < 300
    except Exception as e:
        print(f"  [API] Load error: {e}")
        return False


def wait_for_model_ready(match_tokens, timeout=LOAD_TIMEOUT) -> bool:
    print(f"  Waiting up to {timeout}s for model to be ready...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        mid = find_model_id(match_tokens)
        if mid:
            print(f"  Model ready: {mid}")
            return True
        time.sleep(10)
    print("  Timed out waiting for model to load.")
    return False


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("Auto-switcher: qwen3_8b → qwen3.5_9b")
    print("=" * 60)
    print(f"Watching: {WATCH_FILE}")
    print(f"Will trigger once CSV row count >= {MIN_ROWS_DONE}")
    print()

    # ── Phase 1: Wait for qwen3_8b test_set_07 to finish ────────────────────
    print("Phase 1: Waiting for qwen3_8b to finish test_set_07...")
    while True:
        rows = count_csv_rows(WATCH_FILE)
        print(f"  [{time.strftime('%H:%M:%S')}] test_set_07__qwen3_8b.csv rows: {rows}/{MIN_ROWS_DONE}")
        if rows >= MIN_ROWS_DONE:
            print("  qwen3_8b evaluation COMPLETE.")
            break
        time.sleep(POLL_INTERVAL)

    # Give a moment for final CSV flush
    time.sleep(10)

    # ── Phase 2: Unload qwen3_8b ─────────────────────────────────────────────
    print("\nPhase 2: Unloading qwen3_8b...")
    qwen8b_id = find_model_id(UNLOAD_MATCH_TOKENS)
    if qwen8b_id:
        unload_model(qwen8b_id)
        time.sleep(5)
    else:
        print("  qwen3_8b not found in loaded models (may already be unloaded).")

    # ── Phase 3: Load qwen3.5_9b ─────────────────────────────────────────────
    print("\nPhase 3: Loading qwen3.5-9b...")
    ok = load_model(LOAD_MODEL_PATH)
    if not ok:
        print("  WARNING: Load API call failed. Will still wait to see if model appears.")

    # ── Phase 4: Wait for model to be ready ──────────────────────────────────
    print("\nPhase 4: Waiting for qwen3.5-9b to be ready...")
    ready = wait_for_model_ready(READY_MATCH_TOKENS, timeout=LOAD_TIMEOUT)
    if not ready:
        print("\nERROR: qwen3.5-9b did not become ready. Aborting.")
        print("Please load the model manually in LM Studio and run run_evaluation_cot.py yourself.")
        sys.exit(1)

    # ── Phase 5: Run the evaluation ───────────────────────────────────────────
    print("\nPhase 5: Starting qwen3.5_9b CoT evaluation...")
    print(f"  Running: {PYTHON} {EVAL_SCRIPT}")
    result = subprocess.run([PYTHON, EVAL_SCRIPT], cwd=os.getcwd())

    print("\n" + "=" * 60)
    if result.returncode == 0:
        print("ALL DONE — qwen3.5_9b evaluation complete!")
    else:
        print(f"Evaluation exited with code {result.returncode}.")
    print("=" * 60)


if __name__ == "__main__":
    main()
