---
base_model: Qwen/Qwen3.5-9B
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen3.5-9B
- lora
- sft
- transformers
- trl
- medical
- mcq
- medmcqa
---

# Qwen3.5-9B MedMCQA LoRA Adapter

A LoRA fine-tuned adapter for **Qwen/Qwen3.5-9B** on the [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) dataset — 182K medical multiple-choice questions covering 21 subjects from Indian medical entrance exams (AIIMS/PG style).

## Model Details

- **Developed by:** James Oon ([@jamezoon](https://huggingface.co/jamezoon)), SUTD MSTR-DAIE Deep Learning Project
- **Model type:** Causal LM with LoRA adapter (PEFT)
- **Base model:** `Qwen/Qwen3.5-9B` (dense, 9B parameters, BF16)
- **Language:** English
- **License:** Follows base model license (Qwen3.5)
- **Adapter size:** ~28 MB (`adapter_model.safetensors`)

## Intended Use

Medical multiple-choice question answering. Given a clinical question and 4 options (A–D), the model selects the correct answer with a step-by-step explanation. Subjects covered include Physiology, Anatomy, Biochemistry, Pathology, Pharmacology, Surgery, Medicine, Dental, Gynaecology, Paediatrics, and more.

**Not intended** for real clinical decision-making. This is a research/educational model.

## How to Get Started

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model_id = "Qwen/Qwen3.5-9B"
adapter_id = "jamezoon/qwen3-5-9b-medmcqa-lora"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter_id)

messages = [
    {"role": "system", "content": "You are a helpful tutor for pre-med students preparing for medical entrance exams. Answer the following multiple choice question by thinking step by step, then give the answer."},
    {"role": "user", "content": (
        "Question: Which of the following is the most common cause of mitral stenosis?\n"
        "Options: A. Rheumatic fever  B. Congenital  C. Infective endocarditis  D. SLE\n"
        "Think step by step. Then respond in the format:\n"
        "Explanation: ...\nAnswer: <one of A, B, C, D>"
    )},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Training Details

### Dataset

- **MedMCQA** — 182,822 training samples, 4,183 validation samples
- 21 medical subjects (Dental, Surgery, Medicine, Pathology, Pharmacology, etc.)
- Each sample: question + 4 options + correct answer (1-indexed) + explanation
- Formatted as chat messages with system/user/assistant roles

### Training Procedure

| Hyperparameter | Value |
|---------------|-------|
| Training steps | 1,000 (max_steps) |
| Epochs | 1 |
| Per-device batch size | 4 |
| Gradient accumulation | 4 (effective batch = 16) |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Warmup steps | 100 |
| Max sequence length | 512 tokens |
| Precision | BF16 |
| Optimizer | AdamW (fused) |
| Max grad norm | 1.0 |

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 16 |
| Alpha (α) | 32 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, out_proj |
| Trainable parameters | 7,077,888 (0.079% of 9B) |
| Bias | none |

### Hardware & Training Time

- **Hardware:** NVIDIA GB10 Grace Blackwell (NVIDIA DGX Spark), 121 GB unified CPU+GPU memory
- **Training duration:** ~50 minutes (1,000 steps at ~3s/step)
- **Total run time including evaluation:** ~10 hours (eval_steps=200 triggered 5 evaluation passes × ~55 min each — caused by GatedDeltaNet inference overhead)
- **Framework:** PyTorch 2.x, HuggingFace Transformers, PEFT 0.18.1, TRL (SFTTrainer)

### Known Technical Issue Fixed During Development

Qwen3.5-9B uses a hybrid **GatedDeltaNet** architecture (linear attention layers interleaved with standard softmax attention). This required a monkey-patch to `transformers.masking_utils` and `modeling_qwen3_5.py` to fix a 5D attention mask shape mismatch (`[batch,1,1,seq,seq]` → `[batch,1,seq,seq]`) that caused runtime errors during training.

Without the CUDA kernel (`causal-conv1d`, `flash-linear-attention`), GatedDeltaNet falls back to a PyTorch implementation running at ~3–5s/step instead of the expected ~0.5s/step. This affected both training and evaluation speed on this hardware.

## Evaluation

### Baseline Comparison (dev split, 4,183 samples)

| Model | Accuracy | Notes |
|-------|----------|-------|
| Qwen3.5-9B zero-shot | 26.6% | Barely above random (25%); format failures common |
| **Qwen3.5-9B + LoRA (this adapter)** | TBD | Evaluation in progress |

Zero-shot accuracy is low primarily due to format non-compliance — the base model frequently fails to output a clean `A/B/C/D` answer in zero-shot settings. Fine-tuning addresses both format adherence and medical knowledge.

### Per-Subject Baseline Accuracy (zero-shot, for reference)

Best subjects: Forensic Medicine (35.8%), Psychiatry (31.3%), Gynaecology (31.7%)
Weakest subjects: Orthopaedics (20.0%), Ophthalmology (19.0%), Radiology (23.2%)

## Citation

If you use this adapter, please cite the MedMCQA dataset:

```bibtex
@inproceedings{pmlr-v174-pal22a,
  title     = {MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering},
  author    = {Pal, Ankit and Umapathi, Logesh Kumar and Sankarasubbu, Malaikannan},
  booktitle = {Proceedings of the Conference on Health, Inference, and Learning},
  year      = {2022},
  publisher = {PMLR}
}
```

### Framework Versions

t
 PEFT 0.18.1
- Transformers (latest as of March 2026)
- TRL (SFTTrainer)
- PyTorch 2.x + CUDA
