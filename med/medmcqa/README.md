---
base_model: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
library_name: pytorch
pipeline_tag: text-classification
tags:
- medical
- mcq
- medmcqa
- bert
- pubmedbert
- question-answering
- multiple-choice
- transformers
datasets:
- openlifescienceai/medmcqa
language:
- en
license: mit
---

# PubMedBERT — MedMCQA Fine-tuned Encoder

A fully fine-tuned **discriminative** model based on **microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext** on the [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) dataset — 182K medical multiple-choice questions covering 21 subjects from Indian medical entrance exams (AIIMS/PG style).

Fine-tuned as part of an SUTD Master's deep learning course project comparing zero-shot, LoRA, and full SFT approaches on medical MCQ benchmarks.

## Model Details

- **Developed by:** James Oon ([@jamezoon](https://huggingface.co/jamezoon)), SUTD MSTR-DAIE Deep Learning Project
- **Model type:** Encoder-only (BERT-style) + linear classification head (discriminative MCQ)
- **Base model:** `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` (110M parameters, 768 hidden dim)
- **Language:** English
- **License:** MIT
- **Artefact layout:** `encoder/` (tokenizer + backbone), `mcqa_head.pt` (linear classifier state dict), `mcqa_metadata.json`

## Intended Use

Medical multiple-choice question answering. Given a clinical question and 4 options (A–D), the model scores each `[context | question + option]` pair independently and selects the highest-scoring option. Subjects covered include Physiology, Anatomy, Biochemistry, Pathology, Pharmacology, Surgery, Medicine, Dental, Gynaecology, Paediatrics, and more.

**Not intended** for real clinical decision-making. This is a research/educational model.

## How It Works

Unlike generative models, this is a **discriminative** approach: each of the 4 options is scored individually by encoding `[question + option_i]` (optionally prepending the context/explanation from the `exp` field), and the option with the highest logit wins.

```
Input:  [CLS] context [SEP] question + option_A [SEP]   → logit_A
        [CLS] context [SEP] question + option_B [SEP]   → logit_B
        [CLS] context [SEP] question + option_C [SEP]   → logit_C
        [CLS] context [SEP] question + option_D [SEP]   → logit_D

Output: softmax([logit_A, logit_B, logit_C, logit_D]) → argmax → predicted option
```

## How to Get Started

```python
import os
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download

REPO_ID = "jamezoon/medmcqa-pubmedbert-mcqa"
LETTERS = ["A", "B", "C", "D"]

local_dir = snapshot_download(REPO_ID)
encoder_dir = os.path.join(local_dir, "encoder")

tokenizer = AutoTokenizer.from_pretrained(encoder_dir)
encoder = AutoModel.from_pretrained(encoder_dir)
encoder.eval()

head = nn.Linear(768, 1)
head.load_state_dict(torch.load(os.path.join(local_dir, "mcqa_head.pt"), map_location="cpu"))
head.eval()


def predict(question: str, options: list, context: str = None, max_len: int = 192):
    pairs = [
        (context, question + " " + opt) if context else question + " " + opt
        for opt in options
    ]
    enc = tokenizer.batch_encode_plus(
        pairs, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt"
    )
    with torch.no_grad():
        pooled = encoder(**enc).pooler_output   # (4, 768)
        logits = head(pooled).squeeze(-1)       # (4,)
    probs = torch.softmax(logits, dim=0)
    pred = int(torch.argmax(probs).item())
    return LETTERS[pred], {L: round(float(p), 3) for L, p in zip(LETTERS, probs)}


question = "Which of the following is the most common cause of mitral stenosis?"
options = ["Rheumatic fever", "Congenital", "Infective endocarditis", "SLE"]
letter, probs = predict(question, options)
print(f"Predicted: {letter}")
print(probs)
```

## Training Details

### Dataset

- **MedMCQA** — 182,822 training samples, 4,183 validation samples (dev set)
- 21 medical subjects (Dental, Surgery, Medicine, Pathology, Pharmacology, etc.)
- Each sample: question + 4 options + correct answer (1-indexed) + explanation
- `use_context=True`: the expert explanation (`exp` field) is prepended as context during both training and inference

### Training Procedure

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 2 (with early stopping, patience = 2) |
| Per-device batch size | 16 |
| Learning rate | 2e-4 |
| Optimizer | AdamW (eps = 1e-8) |
| Max sequence length | 192 tokens |
| Hidden dropout | 0.4 |
| Early stopping monitor | `val_loss` (min) |
| Training loss | CrossEntropyLoss over 4-way option logits |

### Model Architecture

| Component | Details |
|-----------|---------|
| Backbone | PubMedBERT-base (12 layers, 768 hidden, 12 heads) |
| Trainable parameters | ~110M (full fine-tuning) |
| Classification head | `Linear(768 → 1)`, trained from scratch |
| Pooling | CLS-token pooler output |

### Hardware & Training Time

- **Hardware:** NVIDIA GPU (CUDA) or Apple Silicon (MPS, detected automatically)
- **Framework:** PyTorch 2.x, HuggingFace Transformers, PyTorch Lightning

## Evaluation

### Training Loss Progression

| Epoch | Val Loss | Val Acc (Lightning) |
|-------|----------|---------------------|
| 0 | 1.267 | 33.2% |
| 1 | 1.250 | 35.3% |
| **Best checkpoint** | **1.23** | **36.0%** |

> Val acc figures above are as logged by the PyTorch Lightning trainer. Due to a known tensor aggregation behaviour in the old `EvalResult` API (logit tensors averaged across batches instead of concatenated), these slightly overstate the true accuracy. See final inference results below.

### Final MCQ Accuracy (Dev Split, 4,183 samples)

Computed from the best-checkpoint predictions saved in `dev_results.csv`:

| Metric | Value |
|--------|-------|
| **Overall accuracy** | **29.74%** (1,244 / 4,183) |
| **Macro-averaged accuracy** | **31.60%** |
| Random-chance baseline | 25.00% |

### Per-Subject Accuracy (Dev Split)

| Subject | Accuracy | n |
|---------|----------|---|
| Forensic Medicine | 43.3% | 67 |
| Radiology | 42.0% | 69 |
| Pediatrics | 37.2% | 234 |
| Physiology | 35.7% | 171 |
| Pharmacology | 34.2% | 243 |
| ENT | 34.0% | 53 |
| Anaesthesia | 32.4% | 34 |
| Social & Preventive Medicine | 31.8% | 129 |
| Gynaecology & Obstetrics | 31.7% | 224 |
| Psychiatry | 31.2% | 16 |
| Pathology | 29.4% | 337 |
| Surgery | 29.0% | 369 |
| Anatomy | 28.2% | 234 |
| Ophthalmology | 27.6% | 58 |
| Dental | 27.4% | 1,318 |
| Biochemistry | 26.9% | 171 |
| Medicine | 26.1% | 295 |
| Orthopaedics | 25.0% | 20 |
| Microbiology | 23.0% | 122 |
| Skin | 17.6% | 17 |

Best subjects: Forensic Medicine (43.3%), Radiology (42.0%). Weakest: Skin (17.6%), Microbiology (23.0%). Dental dominates the eval set (1,318 / 4,183 = 31.5%), and its 27.4% accuracy pulls the overall figure close to random chance. The model beats random chance (25%) in 16 of 20 subjects.

### MCQ Accuracy Comparison (Dev Split, 4,183 samples)

| Model | Overall Acc | Macro Acc | Notes |
|-------|-------------|-----------|-------|
| **PubMedBERT SFT (this model)** | **29.7%** | **31.6%** | Discriminative encoder, argmax over 4 logits |
| Gemma-3-4B-IT zero-shot | TBD | TBD | Generative baseline pending |
| Gemma-3-4B-IT + LoRA | 45.4% | 43.3% | Generative, 4B params |

> **Note on comparability:** PubMedBERT accuracy is computed as direct argmax accuracy over 4 option logits (discriminative). The generative LoRA models produce free-form text and accuracy is measured by extracting the final `A/B/C/D` answer. Token accuracy reported during LoRA training (77–79%) is a next-token prediction metric and is not directly comparable to MCQ answer accuracy.

## Comparison with Other MedMCQA Models in this Project

| | **This model (PubMedBERT)** | [Gemma-3-4B adapter](https://huggingface.co/jamezoon/gemma-3-4b-it-medmcqa-lora) | [Qwen3.5-9B adapter](https://huggingface.co/jamezoon/qwen3-5-9b-medmcqa-lora) | [Qwen3-14B adapter](https://huggingface.co/jamezoon/qwen3-14b-medmcqa-lora) |
|--|--|--|--|--|
| Base model params | **110M** | 4B | 9B | 14B |
| Approach | Discriminative (full SFT) | Generative (LoRA) | Generative (LoRA) | Generative (LoRA) |
| Architecture | Encoder-only (BERT) | Dense transformer | Hybrid (GatedDeltaNet) | Standard transformer |
| Trainable params | ~110M (100%) | 12.9M (0.299%) | 7.1M (0.079%) | 21.0M (0.142%) |
| Best val loss | 1.23 (CE, 4-way) | 1.094 (NLL, generative) | 0.9669 (NLL) | 0.9649 (NLL) |
| Val MCQ acc (overall) | 29.7% | **45.4%** | TBD | TBD |
| Val MCQ acc (macro) | 31.6% | **43.3%** | TBD | TBD |

## Citation

If you use this model, please cite the MedMCQA dataset:

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

- PyTorch 2.x
- Transformers (latest as of March 2026)
- PyTorch Lightning
