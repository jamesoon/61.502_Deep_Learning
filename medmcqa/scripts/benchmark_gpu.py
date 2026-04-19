"""
GPU training benchmark — tests RTX 4080 speed before full training run.
No model download needed. Runs in ~30 seconds.

Usage:
    python scripts/benchmark_gpu.py
"""

import time
import torch
import torch.nn as nn

# ── 1. GPU Info ────────────────────────────────────────────────────────────────
print("=" * 55)
print("GPU BENCHMARK")
print("=" * 55)

if not torch.cuda.is_available():
    print("ERROR: No CUDA GPU detected.")
    raise SystemExit(1)

device = torch.device("cuda")
props = torch.cuda.get_device_properties(0)
print(f"GPU:      {props.name}")
print(f"VRAM:     {props.total_memory / 1e9:.1f} GB")
print(f"PyTorch:  {torch.__version__}")
print()

# ── 2. Memory bandwidth test ───────────────────────────────────────────────────
print("[1/3] Memory bandwidth test...")
size = 1024 * 1024 * 256  # 1 GB of float32
a = torch.randn(size, device=device, dtype=torch.float32)
b = torch.empty_like(a)

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    b.copy_(a)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

gb_moved = 2 * a.nbytes * 10 / 1e9  # read + write × 10 iters
bw = gb_moved / elapsed
print(f"  Bandwidth: {bw:.0f} GB/s  (RTX 4080 peak ~717 GB/s)")
del a, b

# ── 3. Matrix multiply throughput (TFLOPS) ────────────────────────────────────
print("\n[2/3] Compute throughput (BF16 matmul)...")
M = 4096
a = torch.randn(M, M, device=device, dtype=torch.bfloat16)
b = torch.randn(M, M, device=device, dtype=torch.bfloat16)

# Warmup
for _ in range(5):
    torch.matmul(a, b)
torch.cuda.synchronize()

iters = 50
t0 = time.perf_counter()
for _ in range(iters):
    torch.matmul(a, b)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

flops = 2 * M ** 3 * iters / elapsed / 1e12
print(f"  Throughput: {flops:.1f} TFLOPS BF16  (RTX 4080 peak ~165 TFLOPS)")
del a, b

# ── 4. Simulated transformer training loop ────────────────────────────────────
print("\n[3/3] Simulated transformer training (forward + backward)...")

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Roughly mimics a single transformer block at 14B param width
        d = 2048
        self.attn = nn.MultiheadAttention(d, num_heads=16, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d),
        )
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.head = nn.Linear(d, 32000)  # vocab size

    def forward(self, x):
        x = self.ln1(x)
        a, _ = self.attn(x, x, x)
        x = x + a
        x = self.ln2(x)
        x = x + self.ff(x)
        return self.head(x)

model = TinyTransformer().to(device, dtype=torch.bfloat16)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
loss_fn = nn.CrossEntropyLoss()

# Simulate batch: batch=4, seq_len=512, hidden=2048
batch, seq_len, d = 4, 512, 2048
x = torch.randn(batch, seq_len, d, device=device, dtype=torch.bfloat16)
labels = torch.randint(0, 32000, (batch, seq_len), device=device)

# Warmup
for _ in range(3):
    out = model(x)
    loss = loss_fn(out.reshape(-1, 32000), labels.reshape(-1))
    loss.backward()
    optimizer.zero_grad()
torch.cuda.synchronize()

steps = 20
t0 = time.perf_counter()
for _ in range(steps):
    optimizer.zero_grad()
    out = model(x)
    loss = loss_fn(out.reshape(-1, 32000), labels.reshape(-1))
    loss.backward()
    optimizer.step()
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

sps = steps / elapsed
print(f"  Speed:  {sps:.2f} steps/sec  ({elapsed/steps:.2f}s per step)")
print(f"  VRAM used: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

# ── 5. bitsandbytes check ─────────────────────────────────────────────────────
print("\n[bonus] Checking bitsandbytes (QLoRA)...")
try:
    import bitsandbytes as bnb
    # Quick 4-bit linear layer test
    layer = bnb.nn.Linear4bit(256, 256, bias=False, quant_type="nf4",
                               compute_dtype=torch.bfloat16).to(device)
    x_test = torch.randn(4, 256, device=device, dtype=torch.bfloat16)
    _ = layer(x_test)
    print(f"  bitsandbytes {bnb.__version__}: 4-bit NF4 QLoRA works ✓")
except ImportError:
    print("  bitsandbytes not installed — run: pip install bitsandbytes")
except Exception as e:
    print(f"  bitsandbytes error: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"Memory bandwidth:  {bw:.0f} GB/s")
print(f"BF16 compute:      {flops:.1f} TFLOPS")
print(f"Training speed:    {sps:.2f} steps/sec ({elapsed/steps:.2f}s/step)")
print()
print("Estimated full epoch (182K samples, batch=16):")
steps_per_epoch = 182822 // 16
eta_hrs = (steps_per_epoch / sps) / 3600
print(f"  ~{steps_per_epoch:,} steps × {elapsed/steps:.2f}s = ~{eta_hrs:.1f} hours")
print("  (actual QLoRA 4-bit will be slower due to quantization overhead)")
print("=" * 55)
