---
base_model: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
library_name: transformers
pipeline_tag: text-classification
tags:
- base_model:finetune:microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
- transformers
- medical
- multiple-choice
- mcq
- medmcqa
- classification
datasets:
- openlifescienceai/medmcqa
language:
- en
license: mit
---

# PubMedBERT — MedMCQA Fine-tuned Encoder

A full fine-tune of **microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext** on the [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) dataset — 182K medical multiple-choice questions covering 21 subjects from Indian medical entrance exams (AIIMS/NEET PG style).

Fine-tuned as part of an SUTD Master's deep learning course project comparing zero-shot, encoder-based, and generative LLM approaches on medical MCQ benchmarks.

## Model Details

- **Developed by:** James Oon ([@jamezoon](https://huggingface.co/jamezoon)), SUTD MSTR-DAIE Deep Learning Project
- **Model type:** BERT encoder + linear 4-way classification head (discriminative, full fine-tune)
- **Base model:** `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` (110M parameters)
- **Language:** English
- **License:** MIT
- **Head size:** 1 × Linear(768 → 1), ~768 trainable head parameters on top of full encoder

## Intended Use

Medical multiple-choice question answering. Given a clinical question and 4 options (A–D), the model scores each option independently and selects the highest-scoring option as the answer. Subjects covered include Physiology, Anatomy, Biochemistry, Pathology, Pharmacology, Surgery, Medicine, Dental, Gynaecology, Paediatrics, and more.

**Not intended** for real clinical decision-making. This is a research/educational model.

## How to Get Started

```python
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

# 1. Load tokenizer + fine-tuned backbone
backbone_dir = "jamezoon/medmcqa-pubmedbert-mcqa"  # or local path to encoder/
tokenizer = AutoTokenizer.from_pretrained(f"{backbone_dir}/encoder")
encoder   = AutoModel.from_pretrained(f"{backbone_dir}/encoder")

# 2. Rebuild the linear scoring head (768 → 1)
linear = nn.Linear(768, 1)
head_weights = torch.load(
    "mcqa_head.pt",   # or hf_hub_download(repo_id=backbone_dir, filename="mcqa_head.pt")
    map_location="cpu",
    weights_only=True,
)
linear.load_state_dict(head_weights)
encoder.eval(); linear.eval()

# 3. Inference — score all 4 options, pick argmax
question = "Which macrolide is active against Mycobacterium leprae?"
options  = ["Azithromycin", "Roxithromycin", "Clarithromycin", "Framycetin"]
context  = ""   # optional expert explanation text (used during training)

pairs = [(context + " " + question + " " + opt).strip() for opt in options]
inputs = tokenizer.batch_encode_plus(
    pairs, truncation=True, padding="max_length", max_length=192, return_tensors="pt"
)
with torch.no_grad():
    pooled = encoder(**inputs).pooler_output   # (4, 768)
    logits = linear(pooled).view(1, 4)         # (1, 4)
    pred   = torch.argmax(logits, dim=-1).item()

print(f"Predicted: {chr(65 + pred)} — {options[pred]}")
# Expected: C — Clarithromycin
```

## Input Format

This is a **discriminative encoder model**, not generative. Each of the 4 answer options is tokenised as a separate `[CLS] <context?> <question> <option> [SEP]` sequence. All 4 are scored in a single batched forward pass; argmax over logits gives the final answer. `context` is the expert explanation field (`exp`) from the MedMCQA dataset, included during training via `--use_context`.

## Training Details

### Dataset

- **MedMCQA** — 182,822 training samples, 6,150 validation samples, 4,183 test samples
- 21 medical subjects (Anatomy, Physiology, Biochemistry, Pathology, Pharmacology, Surgery, Medicine, etc.)
- Each sample: question + 4 options + 1-indexed correct answer + expert explanation
- Expert explanation (`exp`) prepended to each `(question, option)` pair as context

### Training Procedure

| Hyperparameter | Value |
|----------------|-------|
| Total epochs configured | 5 |
| Epochs completed | 2 (early-stopping patience 2 on val loss) |
| Steps per epoch | ~11,689 |
| Per-device batch size | 16 |
| Learning rate | 2e-4 (AdamW, ε = 1e-8) |
| Max sequence length | 192 tokens |
| Hidden dropout | 0.4 |
| Precision | FP32 |
| Optimizer | AdamW |
| Gradient checkpointing | No |

### Architecture

- BERT encoder: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` — 110M params, 12 layers, hidden size 768, pre-trained on PubMed abstracts + full-text
- Pooled `[CLS]` token representation (768-dim)
- Dropout(0.4) → Linear(768, 1) → reshape to (batch, 4) → CrossEntropyLoss

### Architecture Note

`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` is a **standard BERT-base encoder** pre-trained exclusively on biomedical text from PubMed (abstracts + full-text articles). Unlike general-purpose BERT/RoBERTa, it was never exposed to general web text before fine-tuning, giving it stronger in-domain vocabulary alignment with clinical and scientific MCQ language. Full fine-tuning (all 110M parameters) is performed — no adapter layers or parameter-efficient methods are used.

### Hardware & Training Time

- **Hardware:** Apple M5 Max (Apple Metal / MPS backend)
- **Training duration:** ~17–20 hours (2 completed epochs, ~23,000 steps at ~1.5–2 s/step on MPS)
- **Framework:** PyTorch 2.x, HuggingFace Transformers 4.49.x, PyTorch Lightning 1.0.8

## Evaluation

### Training Loss Progression

| Epoch | Step | Train Loss (step) |
|-------|------|-------------------|
| 0 | 49 | 1.415 |
| 0 | 499 | 1.333 |
| 0 | 2,999 | 1.408 |
| 0 | 5,699 | 1.368 |
| 0 | 8,499 | 1.335 |
| 0 | 11,099 | 1.330 |
| **0 (epoch end)** | **11,426** | **avg 1.329** |
| **1 (epoch end)** | **22,853** | **avg 1.213** |

### Validation Results (Dev Set, 6,150 samples)

| Epoch | Step | Val Loss | Val Accuracy |
|-------|------|----------|--------------|
| 0 | 11,426 | 1.267 | 33.2% |
| **1** | **22,853** | **1.250** | **35.3%** |

Validation loss improved consistently across both completed epochs. Training was halted after epoch 1 due to early-stopping. Random chance baseline for 4-way MCQ is 25.0%.

### MCQ Accuracy Comparison (Dev Split)

| Model | Accuracy | Notes |
|-------|----------|-------|
| PubMedBERT zero-shot | ~25% | 4-way random baseline |
| **PubMedBERT + full fine-tune (this model)** | **35.3%** | 2 completed epochs |

## Comparison with Generative LoRA Adapters (Same Project)

| | This model (PubMedBERT encoder) | [Gemma-3-4B LoRA](https://huggingface.co/jamezoon/gemma-3-4b-it-medmcqa-lora) | [Qwen3-14B LoRA](https://huggingface.co/jamezoon/qwen3-14b-medmcqa-lora) |
|-|-------------------------------|-------------------------------|-------------------------------|
| Base model params | 110M | 4B | 14B |
| Architecture | BERT encoder + linear head | Decoder LLM + LoRA | Decoder LLM + LoRA |
| Task formulation | Discriminative 4-way classification | Generative MCQ with reasoning | Generative MCQ with reasoning |
| Best eval loss | 1.250 | 1.094 | **0.9649** |
| Best token / val acc | 35.3% | 77.8% | **79.20%** |
| Training time | ~17–20 h (Apple M5 Max, MPS) | ~30–40 min (DGX Spark GB10) | ~26 h (DGX Spark GB10) |
| Produces explanations | No | Yes (step-by-step) | Yes (step-by-step) |
| Adapter / model size | ~440 MB (full encoder) | 49 MB adapter | 81 MB adapter |

> **Note:** The lower accuracy of this encoder model vs. the generative LoRA adapters is expected. Encoder-only BERT-style models score all options independently — they cannot reason across options or leverage chain-of-thought. Generative 4B–14B LLMs trained with SFT on full answer+explanation sequences carry far more parametric medical knowledge and can reason before answering.

## Citation

If you use this model, please cite MedMCQA:

```bibtex
@InProceedings{pmlr-v174-pal22a,
  title     = {MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering},
  author    = {Pal, Ankit and Umapathi, Logesh Kumar and Sankarasubbu, Malaikannan},
  booktitle = {Proceedings of the Conference on Health, Inference, and Learning},
  pages     = {248--260},
  year      = {2022},
  publisher = {PMLR}
}
```

## Disclaimer

This model is intended for **research and educational use only**. It is **not suitable** for real clinical decision-making or medical advice. Always consult a qualified medical professional.

---

### Framework Versions

- Transformers 4.49.x
- PyTorch Lightning 1.0.8
- PyTorch 2.2.x
- Python 3.9
