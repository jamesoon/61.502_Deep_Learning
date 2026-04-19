"""
MedMCQA cross-encoder training (DeBERTa-v3-large by default).

Usage:
    # DeBERTa-v3-large (recommended)
    python3 train.py --model microsoft/deberta-v3-large --dataset_folder_name data --use_context

    # BioLinkBERT-large (biomedical domain)
    python3 train.py --model michiyasunaga/BioLinkBERT-large --dataset_folder_name data --use_context

    # Quick smoke test (10 steps)
    python3 train.py --max_steps 10 --dataset_folder_name data

    # With JSON data (from medmcqa/ processed data)
    python3 train.py --train_file ../medmcqa/data/train.json --dev_file ../medmcqa/data/dev.json
"""

import argparse
import csv
import json
import os
import time

# ── GB10 Blackwell nvrtc fix ──────────────────────────────────────────────────
# Must be set BEFORE torch is imported. PYTORCH_JIT=0 disables the NNC/texpr
# eager-mode fuser that tries to JIT-compile DeBERTa's make_log_bucket_position
# ops (lt/gt/log/where) via nvrtc — which fails because GB10 Blackwell's arch
# code is not yet recognised by the nvrtc bundled with this PyTorch build.
os.environ.setdefault("PYTORCH_JIT", "0")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import pandas as pd
import torch
import torch._dynamo
# Belt-and-suspenders: also disable fusers via Python API after import.
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()
torch.jit.enable_onednn_fusion(False)
torch._C._jit_set_nvfuser_enabled(False) if hasattr(torch._C, '_jit_set_nvfuser_enabled') else None
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_profiling_executor(False)
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

# ── GB10 Blackwell: patch DeBERTa relative-position bucketing ─────────────────
# Even with all JIT flags disabled, PyTorch's NNC/texpr eager-mode fuser still
# tries to compile a fused lt/gt/log/where kernel for make_log_bucket_position
# via nvrtc — which fails on GB10 Blackwell ("invalid value for --gpu-arch").
# The position bucket tensors are at most seq_len×seq_len integers (~65K elems);
# computing them on CPU and transferring back adds negligible overhead.
try:
    import transformers.models.deberta_v2.modeling_deberta_v2 as _deberta_v2
    _orig_make_log_bucket = _deberta_v2.make_log_bucket_position

    def _cpu_make_log_bucket_position(relative_pos, bucket_size, max_position):
        if relative_pos.device.type == "cuda":
            return _orig_make_log_bucket(
                relative_pos.cpu(), bucket_size, max_position
            ).to(relative_pos.device)
        return _orig_make_log_bucket(relative_pos, bucket_size, max_position)

    _deberta_v2.make_log_bucket_position = _cpu_make_log_bucket_position
    print("[GB10 Fix] Patched DeBERTa make_log_bucket_position → CPU fallback.")
except Exception as _patch_err:
    print(f"[GB10 Fix] Could not patch make_log_bucket_position: {_patch_err}")

from conf.args import Arguments
from dataset import MCQADataset
from model import MCQAModel

DEFAULT_DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ── Dataset resolution ─────────────────────────────────────────────────────────

def resolve_dataset_folder(name: str) -> str:
    candidates = []
    if os.path.isabs(name):
        candidates.append(name)
    else:
        candidates.append(os.path.abspath(name))
        candidates.append(os.path.join(DEFAULT_DATASET_ROOT, name))
        candidates.append(DEFAULT_DATASET_ROOT)

    for path in candidates:
        # Check for CSV or JSON data files
        has_train = (os.path.exists(os.path.join(path, "train.csv")) or
                     os.path.exists(os.path.join(path, "train.json")))
        has_dev = (os.path.exists(os.path.join(path, "dev.csv")) or
                   os.path.exists(os.path.join(path, "validation.csv")) or
                   os.path.exists(os.path.join(path, "dev.json")))
        has_test = (os.path.exists(os.path.join(path, "test.csv")) or
                    os.path.exists(os.path.join(path, "test.json")))
        if has_train and has_dev:
            return path

    raise FileNotFoundError(
        f"Cannot find dataset folder with train/dev data. Checked: {candidates}"
    )


def find_data_file(folder: str, base: str) -> str:
    """Find data file, preferring CSV then JSON."""
    for ext in (".csv", ".json"):
        path = os.path.join(folder, base + ext)
        if os.path.exists(path):
            return path
    # Special case: dev might be called validation
    if base == "dev":
        for ext in (".csv", ".json"):
            path = os.path.join(folder, "validation" + ext)
            if os.path.exists(path):
                return path
    raise FileNotFoundError(f"No {base}.csv/.json in {folder}")


# ── Train / eval loops ─────────────────────────────────────────────────────────

def run_epoch(model: MCQAModel, dataloader, optimizer, device: str,
              scheduler=None, max_grad_norm: float = 1.0,
              grad_accum_steps: int = 1,
              max_steps: int | None = None,
              desc: str = "Train") -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, total = 0.0, 0, 0
    steps = 0
    micro_step = 0

    # DeBERTa-v3 keeps some internal weights in FP16, which causes NaN with
    # mixed-precision autocast (both FP16 and BF16).  At 434M params it fits
    # fine in FP32, so we skip autocast entirely.

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        pbar = tqdm(dataloader, desc=desc, leave=False)
        if is_train:
            optimizer.zero_grad()
        for inputs, labels in pbar:
            for k in inputs:
                inputs[k] = inputs[k].to(device)
            labels = labels.to(device)

            logits = model(**inputs)
            loss = model.ce_loss(logits, labels)

            # Track stats before backward so they're always updated
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if is_train:
                (loss / grad_accum_steps).backward()
                micro_step += 1

                if micro_step % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                    steps += 1
                    pbar.set_postfix(loss=f"{loss.item():.4f}",
                                     acc=f"{correct/total:.4f}")
                    if max_steps and steps >= max_steps:
                        break
            else:
                steps += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 acc=f"{correct/total:.4f}")

        # Handle leftover micro-steps after the loop ends
        if is_train and micro_step % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            steps += 1

    return total_loss / max(steps, 1), correct / max(total, 1)


def run_inference(model: MCQAModel, dataloader, device: str) -> list[int]:
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="Inference", leave=False):
            for k in inputs:
                inputs[k] = inputs[k].to(device)
            logits = model(**inputs)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            predictions.extend(preds)
    return predictions


# ── HF export ─────────────────────────────────────────────────────────────────

def export_hf_artifacts(model: MCQAModel, model_name: str,
                        export_root: str, experiment_name: str):
    export_dir = os.path.join(export_root, experiment_name.replace("/", "_"))
    encoder_dir = os.path.join(export_dir, "encoder")
    os.makedirs(encoder_dir, exist_ok=True)

    model.tokenizer.save_pretrained(encoder_dir)
    model.model.save_pretrained(encoder_dir)

    torch.save(model.head.state_dict(), os.path.join(export_dir, "mcqa_head.pt"))

    meta = {
        "task": "multiple_choice_qa",
        "architecture": "cross_encoder",
        "base_model_name": model_name,
        "custom_head_type": "2layer_mlp",
        "head_hidden_dim": model.args.get("mlp_hidden", 256),
        "num_choices": model.args.get("num_choices", 4),
        "label_smoothing": model.args.get("label_smoothing", 0.1),
        "use_context": model.args.get("use_context", True),
        "max_len": model.args.get("max_len", 512),
        "encoder_subdir": "encoder",
        "custom_head_weights": "mcqa_head.pt",
    }
    with open(os.path.join(export_dir, "mcqa_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"HF artifacts exported to {export_dir}")


# ── CSV metric logger ──────────────────────────────────────────────────────────

class CSVMetricLogger:
    def __init__(self, path: str):
        self.path = path
        self._rows = []

    def log(self, row: dict):
        self._rows.append(row)
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            for r in self._rows:
                writer.writerow(r)


# ── Main training ──────────────────────────────────────────────────────────────

def train(model_name: str, args: Arguments, experiment_name: str,
          models_folder: str, max_steps: int | None = None):

    device = "cuda" if torch.cuda.is_available() else \
             ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Train] device={device}  model={model_name}")
    print(f"[Train] max_len={args.max_len}  batch_size={args.batch_size}  "
          f"lr={args.learning_rate}  label_smoothing={args.label_smoothing}")

    torch.manual_seed(42)

    # Datasets
    train_dataset = MCQADataset(args.train_csv, args.use_context)
    val_dataset   = MCQADataset(args.dev_csv,   args.use_context)
    print(f"[Train] train={len(train_dataset)}  val={len(val_dataset)}")

    # Model — force FP32 to avoid NaN from DeBERTa's internal FP16 weights
    model = MCQAModel(model_name, args.__dict__).float().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] hidden_size={model.args['hidden_size']}  "
          f"params={total_params/1e6:.1f}M  trainable={trainable_params/1e6:.1f}M")

    # Optimizer with weight decay (exclude bias and LayerNorm)
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                     if not any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                     if any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(param_groups, lr=args.learning_rate, eps=1e-8)

    # Dataloaders
    train_dl = model.get_train_dataloader(train_dataset)
    val_dl   = model.get_val_dataloader(val_dataset)

    # Scheduler: linear warmup then linear decay
    # total_steps = optimizer steps (not micro-batches)
    grad_accum = args.grad_accum_steps
    steps_per_epoch = len(train_dl) // grad_accum
    total_steps = steps_per_epoch * args.num_epochs
    if max_steps:
        total_steps = min(total_steps, max_steps * args.num_epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"[Train] batch={args.batch_size}x{grad_accum} (eff={args.batch_size * grad_accum})  "
          f"steps/epoch={steps_per_epoch}  total_steps={total_steps}  warmup={warmup_steps}")

    # Logging / checkpointing
    exp_dir = os.path.join(models_folder, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    best_ckpt = os.path.join(exp_dir, "best.pt")
    logger = CSVMetricLogger(os.path.join(exp_dir, "metrics.csv"))

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(args.num_epochs):
        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model, train_dl, optimizer, device,
            scheduler=scheduler,
            grad_accum_steps=grad_accum,
            max_steps=max_steps,
            desc=f"Epoch {epoch} train",
        )
        val_loss, val_acc = run_epoch(
            model, val_dl, None, device,
            desc=f"Epoch {epoch} val",
        )
        elapsed = time.time() - t0

        lr_now = scheduler.get_last_lr()[0]
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
              f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
              f"  lr={lr_now:.2e}  ({elapsed:.0f}s)")

        logger.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                    "val_loss": val_loss, "val_acc": val_acc, "lr": lr_now,
                    "time_s": elapsed})

        # Track best by accuracy (more stable than loss with label smoothing)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            model.save_checkpoint(best_ckpt, model_name, epoch, val_loss, val_acc)
            print(f"  -> New best val_acc={val_acc:.4f} — checkpoint saved.")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print("[Train] Early stopping triggered.")
                break

    # Load best checkpoint for final evaluation
    print(f"\n[Inference] Loading best checkpoint from {best_ckpt}")
    best_model = MCQAModel.load_checkpoint(best_ckpt, device=device).to(device)
    best_model.eval()

    # Dev predictions
    val_preds = run_inference(best_model, val_dl, device)
    val_src = pd.read_csv(args.dev_csv) if args.dev_csv.endswith(".csv") else \
              pd.DataFrame(json.load(open(args.dev_csv))) if args.dev_csv.endswith(".json") else \
              pd.read_json(args.dev_csv, lines=True)
    val_src["predictions"] = [p + 1 for p in val_preds]
    results_path = os.path.join(exp_dir, "dev_results.csv")
    val_src.to_csv(results_path, index=False)
    print(f"Dev predictions -> {results_path}")

    # Compute final dev accuracy
    if "cop" in val_src.columns:
        correct = (val_src["predictions"] == val_src["cop"]).sum()
        total = len(val_src)
        print(f"[Final] Dev accuracy: {correct}/{total} = {correct/total:.4f}")

    # Test predictions (if available)
    if hasattr(args, "test_csv") and args.test_csv and os.path.exists(args.test_csv):
        test_dataset = MCQADataset(args.test_csv, args.use_context)
        test_dl = best_model.get_val_dataloader(test_dataset)
        test_preds = run_inference(best_model, test_dl, device)
        if args.test_csv.endswith(".csv"):
            test_src = pd.read_csv(args.test_csv)
        elif args.test_csv.endswith(".json"):
            test_src = pd.DataFrame(json.load(open(args.test_csv)))
        else:
            test_src = pd.read_json(args.test_csv, lines=True)
        test_src["predictions"] = [p + 1 for p in test_preds]
        test_path = os.path.join(exp_dir, "test_results.csv")
        test_src.to_csv(test_path, index=False)
        print(f"Test predictions -> {test_path}")

    # HF export
    export_hf_artifacts(best_model, model_name, "./hf_export", experiment_name)

    return best_val_acc


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torch.optim import AdamW  # ensure in scope

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/deberta-v3-large")
    parser.add_argument("--dataset_folder_name", default="data")
    parser.add_argument("--train_file", default=None, help="Explicit path to train data (overrides folder)")
    parser.add_argument("--dev_file", default=None, help="Explicit path to dev data (overrides folder)")
    parser.add_argument("--test_file", default=None, help="Explicit path to test data (overrides folder)")
    parser.add_argument("--use_context", default=False, action="store_true")
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--max_len", type=int, default=None, help="Override max sequence length")
    parser.add_argument("--grad_accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--freeze_layers", type=int, default=None, help="Freeze bottom N encoder layers (default 18)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Stop after this many training steps per epoch (for smoke tests)")
    cmd = parser.parse_args()

    # Resolve data paths
    if cmd.train_file and cmd.dev_file:
        train_path = cmd.train_file
        dev_path = cmd.dev_file
        test_path = cmd.test_file or ""
    else:
        folder = resolve_dataset_folder(cmd.dataset_folder_name)
        train_path = find_data_file(folder, "train")
        dev_path = find_data_file(folder, "dev")
        try:
            test_path = find_data_file(folder, "test")
        except FileNotFoundError:
            test_path = ""

    args = Arguments(
        train_csv=train_path,
        test_csv=test_path,
        dev_csv=dev_path,
        pretrained_model_name=cmd.model,
        use_context=cmd.use_context,
    )
    if cmd.num_epochs is not None:
        args.num_epochs = cmd.num_epochs
    if cmd.batch_size is not None:
        args.batch_size = cmd.batch_size
    if cmd.lr is not None:
        args.learning_rate = cmd.lr
    if cmd.max_len is not None:
        args.max_len = cmd.max_len
    if cmd.grad_accum is not None:
        args.grad_accum_steps = cmd.grad_accum
    if cmd.freeze_layers is not None:
        args.freeze_layers = cmd.freeze_layers

    exp_name = (
        f"{cmd.model}@@@cross_encoder"
        f"@@@use_context{cmd.use_context}@@@seqlen{args.max_len}"
    ).replace("/", "_")

    train(
        model_name=cmd.model,
        args=args,
        experiment_name=exp_name,
        models_folder="./models",
        max_steps=cmd.max_steps,
    )
