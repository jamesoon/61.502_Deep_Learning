---
base_model: google/gemma-3-4b-it
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:google/gemma-3-4b-it
- lora
- sft
- transformers
- trl
- medical
- mcq
- medmcqa
datasets:
- openlifescienceai/medmcqa
language:
- en
license: gemma
---

# Gemma 3 4B — MedMCQA LoRA Adapter

A LoRA fine-tuned adapter for **google/gemma-3-4b-it** on the [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) dataset — 182K medical multiple-choice questions covering 21 subjects from Indian medical entrance exams (AIIMS/PG style).

Fine-tuned as part of an SUTD Master's deep learning course project comparing zero-shot, LoRA, and full SFT approaches on medical MCQ benchmarks.

## Model Details

- **Developed by:** James Oon ([@jamezoon](https://huggingface.co/jamezoon)), SUTD MSTR-DAIE Deep Learning Project
- **Model type:** Causal LM with LoRA adapter (PEFT)
- **Base model:** `google/gemma-3-4b-it` (dense, 4B parameters, standard transformer)
- **Language:** English
- **License:** Follows base model license ([Gemma](https://ai.google.dev/gemma/terms))
- **Adapter size:** ~49 MB (`adapter_model.safetensors`)

## Intended Use

Medical multiple-choice question answering. Given a clinical question and 4 options (A–D), the model selects the correct answer with a step-by-step explanation. Subjects covered include Physiology, Anatomy, Biochemistry, Pathology, Pharmacology, Surgery, Medicine, Dental, Gynaecology, Paediatrics, and more.

**Not intended** for real clinical decision-making. This is a research/educational model.

## How to Get Started

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_id = "google/gemma-3-4b-it"
adapter_id = "jamezoon/gemma-3-4b-it-medmcqa-lora"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter_id)
model.eval()

messages = [
    {"role": "system", "content": (
        "You are a helpful tutor for pre-med students preparing for medical entrance exams. "
        "Answer the following multiple choice question by thinking step by step, then give the answer."
    )},
    {"role": "user", "content": (
        "Question: Which of the following is the most common cause of mitral stenosis?\n"
        "Options: A. Rheumatic fever  B. Congenital  C. Infective endocarditis  D. SLE\n"
        "Think step by step. Then respond in the format:\n"
        "Explanation: ...\nAnswer: <one of A, B, C, D>"
    )},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Prompt Format

The model expects chat-template format with the following structure:

```
System: You are a helpful tutor for pre-med students preparing for medical entrance exams.
        You answer multiple-choice questions with step-by-step reasoning.

User:   Question: {question}
        Options:
        A. {option_a}
        B. {option_b}
        C. {option_c}
        D. {option_d}

        Think step by step. Then respond in the format:
        Explanation: ...
        Answer: <one of A, B, C, D>
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
| Training steps | 1,200 (max_steps — ~10.5% of 1 full epoch) |
| Epochs | 1 (partial) |
| Per-device batch size | 2 |
| Gradient accumulation | 8 (effective batch = 16) |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Warmup steps | 200 |
| Max sequence length | 512 tokens |
| Precision | BF16 |
| Optimizer | AdamW |
| Gradient checkpointing | Enabled |

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 16 |
| Alpha (α) | 32 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, out_proj |
| Trainable parameters | 12,894,208 (0.299% of 4.3B) |
| Bias | none |

### Hardware & Training Time

- **Hardware:** NVIDIA GB10 Grace Blackwell (NVIDIA DGX Spark), 121 GB unified CPU+GPU memory
- **Training duration:** ~30–40 minutes (1,200 steps)
- **Framework:** PyTorch 2.x, HuggingFace Transformers, PEFT 0.18.1, TRL (SFTTrainer)

### Architecture Note

`google/gemma-3-4b-it` is a **standard dense transformer** with full compatibility with BF16 training and standard CUDA kernels. No special architecture patches were required.

## Evaluation

### Training Loss Progression

| Step | Train Loss | Token Accuracy |
|------|-----------|----------------|
| 10   | 5.164     | 48.3%          |
| 50   | 2.131     | 61.7%          |
| 100  | ~1.39     | ~73%           |
| 200  | ~1.27     | ~74%           |
| 1200 | 1.27      | 74.4%          |

### Validation Results (Dev Set, 4,183 samples)

| Checkpoint | Eval Loss | Token Accuracy |
|-----------|-----------|----------------|
| Step 200  | 1.143     | 77.3%          |
| Step 400  | 1.125     | 77.5%          |
| Step 600  | 1.111     | 77.7%          |
| Step 800  | 1.106     | 77.7%          |
| Step 1000 | 1.105     | 77.7%          |
| **Step 1200 (best)** | **1.094** | **77.8%** |

Eval loss was still decreasing at step 1200 — the model had not yet fully converged.

### MCQ Accuracy Comparison (Dev Split, 4,183 samples)

| Model | Overall Acc | Macro Acc | Notes |
|-------|-------------|-----------|-------|
| Gemma-3-4B-IT zero-shot | TBD | TBD | Baseline evaluation pending |
| **Gemma-3-4B-IT + LoRA (this adapter)** | **45.4%** | **43.3%** | 1,900 / 4,183 correct |

Format compliance: only 33 / 4,183 responses (0.8%) failed to produce a valid A/B/C/D answer — a significant improvement over zero-shot format failures seen in larger models.

### Per-Subject MCQ Accuracy (LoRA adapter, dev split)

| Subject | Accuracy | n |
|---------|----------|---|
| Radiology | 53.6% | 69 |
| Physiology | 53.2% | 171 |
| Biochemistry | 53.2% | 171 |
| Pharmacology | 51.0% | 243 |
| Microbiology | 50.8% | 122 |
| Medicine | 47.8% | 295 |
| Pathology | 47.2% | 337 |
| ENT | 47.2% | 53 |
| Anatomy | 47.0% | 234 |
| Gynaecology & Obstetrics | 46.0% | 224 |
| Ophthalmology | 44.8% | 58 |
| Social & Preventive Medicine | 44.2% | 129 |
| Psychiatry | 43.8% | 16 |
| Surgery | 43.6% | 369 |
| Pediatrics | 42.3% | 234 |
| Dental | 42.1% | 1,318 |
| Orthopaedics | 40.0% | 20 |
| Anaesthesia | 38.2% | 34 |
| Forensic Medicine | 37.3% | 67 |
| Skin | 35.3% | 17 |

Best subjects: Radiology (53.6%), Physiology (53.2%), Biochemistry (53.2%). Weakest: Skin (35.3%), Forensic Medicine (37.3%). Dental dominates sample count (1,318 / 4,183 = 31.5% of eval), so its 42.1% pulls down the overall figure.

## Comparison with Other MedMCQA Adapters

| | [Qwen3.5-9B adapter](https://huggingface.co/jamezoon/qwen3-5-9b-medmcqa-lora) | [Qwen3-14B adapter](https://huggingface.co/jamezoon/qwen3-14b-medmcqa-lora) | This adapter (Gemma-3-4B) |
|--|----------------|-----------------|-----------------|
| Base model params | 9B | 14B | **4B** |
| Architecture | Hybrid (GatedDeltaNet) | Standard transformer | Standard transformer |
| Trainable params | 7.1M (0.079%) | 21.0M (0.142%) | 12.9M (0.299%) |
| Best eval loss | 0.9669 | 0.9649 | 1.094 |
| Best token acc | 78.7% | 79.20% | **77.8%** |
| MCQ accuracy (overall) | TBD | TBD | **45.4%** |
| MCQ accuracy (macro) | TBD | TBD | **43.3%** |
| Adapter size | 28 MB | 81 MB | **49 MB** |

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

- PEFT 0.18.1
- Transformers (latest as of March 2026)
- TRL (SFTTrainer)
- PyTorch 2.x + CUDA