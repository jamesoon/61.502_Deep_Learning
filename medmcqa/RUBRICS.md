# Best Practice Rubrics for MedMCQA Fine-Tuning

## 1. Data Quality

| Criterion | Standard |
|-----------|----------|
| No data leakage | Test set never seen during training or validation-based early stopping |
| Label correctness | `cop` field maps 1→A, 2→B, 3→C, 4→D consistently |
| Train/val/test split | Use the provided splits — do not re-split (leaks subject distribution) |
| Explanation quality | MedMCQA `exp` field is noisy; note in report if using explanations in supervision |
| Sample cap | When using `--max_train N`, sample randomly with a fixed seed for reproducibility |

## 2. Training Stability

| Criterion | Recommendation |
|-----------|----------------|
| Gradient clipping | `max_grad_norm=1.0` — catch early instability |
| Learning rate | SFT: 2e-5; QLoRA: 2e-4. Monitor loss curve; reduce if spiky |
| Warmup | 3% warmup ratio with cosine decay |
| Early stopping | Stop on `eval_loss` with patience=3 checkpoints |
| BF16 | Preferred over FP16 on Blackwell (better dynamic range, avoids NaN) |
| Log every 10 steps | Check WandB for loss divergence in first 100 steps |

## 3. Evaluation Rigor

| Criterion | Standard |
|-----------|----------|
| Primary metric | Multiple-choice accuracy (letter extraction, not embedding similarity) |
| Answer extraction | Use regex `Answer:\s*([ABCD])` — report parse failure rate separately |
| Per-subject accuracy | Report all 21 MedMCQA subjects; compute macro-average to avoid size bias |
| Comparison | Always include zero-shot baseline run for apples-to-apples comparison |
| Test set discipline | Run test evaluation **once** at the very end; use dev set for all tuning |
| Sample size | Evaluate on full test set (4,183 samples) for final numbers |

## 4. Comparison Methodology (Baseline vs SFT vs QLoRA)

| Configuration | Expected Accuracy Range | Notes |
|---------------|------------------------|-------|
| Zero-shot (Qwen3-30B-A3B) | 55–70% | Strong general reasoning |
| QLoRA fine-tuned | +5–15% gain | Primary contribution |
| Full SFT fine-tuned | +5–12% gain | May overfit with small data; watch val loss |

- Report both overall and macro-averaged accuracy
- Include 95% confidence intervals: `±1.96 * sqrt(p*(1-p)/n)`
- Qualitative error analysis: manually review 50 failure cases and categorize:
  - Knowledge gaps (model lacks specific medical fact)
  - Reasoning errors (model has knowledge but wrong logic)
  - Answer format issues (model output not parseable)
  - Ambiguous questions

## 5. Reproducibility Checklist

- [ ] Fixed random seed (`--seed 42`) for data shuffling
- [ ] All hyperparameters logged to WandB
- [ ] Requirements pinned in `requirements_training.txt`
- [ ] Model checkpoint uploaded (HuggingFace Hub or Google Drive)
- [ ] `data_prep.py` output deterministic given same seed
- [ ] Eval script standalone (no training code dependency)
- [ ] README documents exact commands to reproduce each number in the paper

## 6. Reporting Template

```
Model            | Overall Acc | Macro-Avg Acc | Params Trained | Train Time
-----------------|-------------|---------------|----------------|----------
Baseline (0-shot)|   0.XXX     |    0.XXX      |       0        |    —
SFT              |   0.XXX     |    0.XXX      |    ~30B        |  ~X hrs
QLoRA (r=16)     |   0.XXX     |    0.XXX      |   ~150M        |  ~X hrs
```

Include:
- Training loss curve (WandB screenshot or matplotlib)
- Validation accuracy curve across checkpoints
- Confusion matrix for each configuration
- Subject-wise heatmap: model × subject accuracy
