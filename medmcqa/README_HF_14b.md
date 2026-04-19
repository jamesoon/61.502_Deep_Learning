---
base_model: Qwen/Qwen3-14B
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen3-14B
- lora
- sft
- transformers
- trl
- medical
- mcq
- medmcqa
---

# Qwen3-14B MedMCQA LoRA Adapter

A LoRA fine-tuned adapter for **Qwen/Qwen3-14B** on the [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) dataset — 182K medical multiple-choice questions covering 21 subjects from Indian medical entrance exams (AIIMS/PG style).

## Model Details

- **Developed by:** James Oon ([@jamezoon](https://huggingface.co/jamezoon)), SUTD MSTR-DAIE Deep Learning Project
- **Model type:** Causal LM with LoRA adapter (PEFT)
- **Base model:** `Qwen/Qwen3-14B` (dense, 14B parameters, BF16, standard transformer)
- **Language:** English
- **License:** Follows base model license (Qwen3)
- **Adapter size:** ~81 MB (`adapter_model.safetensors`)

## Intended Use

Medical multiple-choice question answering. Given a clinical question and 4 options (A–D), the model selects the correct answer with a step-by-step explanation. Subjects covered include Physiology, Anatomy, Biochemistry, Pathology, Pharmacology, Surgery, Medicine, Dental, Gynaecology, Paediatrics, and more.

**Not intended** for real clinical decision-making. This is a research/educational model.

## How to Get Started

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model_id = "Qwen/Qwen3-14B"
adapter_id = "jamezoon/qwen3-14b-medmcqa-lora"

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
| Training steps | 1,000 (max_steps — ~8.7% of 1 full epoch) |
| Epochs | 1 (partial) |
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
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Trainable parameters | 20,971,520 (0.1418% of 14B) |
| Bias | none |

### Hardware & Training Time

- **Hardware:** NVIDIA GB10 Grace Blackwell (NVIDIA DGX Spark), 121 GB unified CPU+GPU memory
- **Training duration:** ~26 hours total (1,000 training steps + 5 evaluation passes × ~95 min each)
- **Actual training steps:** ~83 minutes (1,000 steps at ~5s/step)
- **Framework:** PyTorch 2.x, HuggingFace Transformers, PEFT 0.18.1, TRL (SFTTrainer)

### Architecture Note

`Qwen/Qwen3-14B` is a **standard dense transformer** (not the Qwen3.5 hybrid variant). It does not use GatedDeltaNet linear attention layers, making it fully compatible with standard CUDA training without special kernel requirements.

## Evaluation

### Training Loss Progression

| Step | Train Loss | Token Accuracy |
|------|-----------|----------------|
| 10 | 2.972 | 55.5% |
| 50 | 1.502 | 69.5% |
| 100 | 1.124 | 76.6% |
| 200 | 1.061 | 77.3% |
| 600 | 1.052 | 77.3% |
| 1,000 | 1.068 | 76.7% |

### Validation Loss (Dev Set, 4,183 samples)

| Checkpoint | Eval Loss | Token Accuracy |
|-----------|-----------|----------------|
| Step 200 | 0.9825 | 78.96% |
| Step 600 | 0.9746 | 79.02% |
| Step 800 | 0.9681 | 79.14% |
| Step 1000 | 0.9664 | 79.18% |
| **Best (saved)** | **0.9649** | **79.20%** |

Eval loss improved consistently throughout training, indicating good generalisation.

### MCQ Accuracy Comparison (Dev Split, 4,183 samples)

| Model | Accuracy | Notes |
|-------|----------|-------|
| Qwen3-14B zero-shot | 27.4% | Format failures common (~12.5% None responses) |
| **Qwen3-14B + LoRA (this adapter)** | TBD | Evaluation in progress |

Zero-shot accuracy is low primarily due to format non-compliance — the base model frequently fails to output a clean `A/B/C/D` answer in zero-shot settings. LoRA fine-tuning addresses both format adherence and domain knowledge.

### Per-Subject Zero-Shot Baseline (for reference)

Best subjects: Anaesthesia (38.2%), Psychiatry (37.5%), Radiology (31.9%)
Weakest subjects: Orthopaedics (10.0%), Skin (11.8%), Anatomy (19.2%)

## Comparison with Qwen3.5-9B Adapter

| | [Qwen3.5-9B adapter](https://huggingface.co/jamezoon/qwen3-5-9b-medmcqa-lora) | This adapter (Qwen3-14B) |
|--|----------------|-----------------|
| Base model params | 9B | 14B |
| Architecture | Hybrid (GatedDeltaNet) | Standard transformer |
| Trainable params | 7.1M (0.079%) | 21.0M (0.142%) |
| Best eval loss | 0.9669 | **0.9649** |
| Best token acc | 78.7% | **79.20%** |
| Adapter size | 28MB | 81MB |

## Citation

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

- PEFT 0.18.1
- Transformers (latest as of March 2026)
- TRL (SFTTrainer)
- PyTorch 2.x + CUDA
