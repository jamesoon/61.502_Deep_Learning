---
base_model: Qwen/Qwen3-14B
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - q4_k_m
  - medical
  - medmcqa
  - mcq
  - llama-cpp
  - ollama
  - lm-studio
license: apache-2.0
language:
  - en
---

# Qwen3-14B MedMCQA — GGUF Q4_K_M

A **Q4_K_M GGUF** quantization of [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) fine-tuned on the [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa) dataset — 182K medical multiple-choice questions covering 21 subjects from Indian medical entrance exams (AIIMS/PG style).

The LoRA adapter ([jamezoon/qwen3-14b-medmcqa-lora](https://huggingface.co/jamezoon/qwen3-14b-medmcqa-lora)) was fully merged into the base weights before quantization. This file is ready for offline inference — no Python ML stack required.

> **Developed by:** James Oon ([@jamezoon](https://huggingface.co/jamezoon)), SUTD MSTR-DAIE Deep Learning Project

---

## Files

| File | Size | Format | Use with |
|------|------|--------|----------|
| `lora-14b-medmcqa-q4_k_m.gguf` | 8.4 GB | GGUF Q4_K_M | llama.cpp, Ollama, LM Studio, Jan |
| `tokenizer.json` | 11 MB | JSON | Reference tokenizer |
| `tokenizer_config.json` | 661 B | JSON | Tokenizer config |

---

## Quick Start

### llama.cpp

```bash
# Download
huggingface-cli download jamezoon/qwen3-14b-medmcqa-gguf \
    lora-14b-medmcqa-q4_k_m.gguf --local-dir ./

# Run (GPU offload — adjust -ngl to match your VRAM)
./llama-cli -m lora-14b-medmcqa-q4_k_m.gguf \
  --ctx-size 2048 -n 256 -ngl 99 \
  -p "Question: A 45-year-old male presents with sudden-onset chest pain radiating to the left arm. Which enzyme is most useful in diagnosing myocardial infarction at 6 hours?\nOptions: A. LDH  B. CK-MB  C. Troponin I  D. AST\nExplanation:"
```

### Ollama

```bash
# Create Modelfile
cat > Modelfile << 'EOF'
FROM ./lora-14b-medmcqa-q4_k_m.gguf
SYSTEM "You are a helpful medical tutor for pre-med students preparing for AIIMS/PG entrance exams. For each question, think step by step, then provide the answer in the format:\nExplanation: ...\nAnswer: <A, B, C, or D>"
EOF

ollama create medmcqa-14b -f Modelfile
ollama run medmcqa-14b
```

### LM Studio

Download the `.gguf` file and load it directly. Recommended settings:
- Context length: 2048
- GPU layers: max available
- Temperature: 0.1 (for deterministic answers)

### Python (llama-cpp-python)

```python
from llama_cpp import Llama

llm = Llama(
    model_path="lora-14b-medmcqa-q4_k_m.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,   # -1 = all layers on GPU; set 0 for CPU-only
    verbose=False,
)

prompt = (
    "<|im_start|>system\n"
    "You are a helpful tutor for pre-med students preparing for medical entrance exams.\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "Question: Which nerve is damaged in wrist drop?\n"
    "Options: A. Radial  B. Ulnar  C. Median  D. Musculocutaneous\n"
    "Think step by step. Then respond in the format:\nExplanation: ...\nAnswer: <one of A, B, C, D>\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

response = llm(prompt, max_tokens=256, stop=["<|im_end|>"])
print(response["choices"][0]["text"])
```

---

## Memory Requirements

| Precision | File Size | Min VRAM / RAM |
|-----------|-----------|----------------|
| BF16 (original) | ~28 GB | ~30 GB |
| **Q4_K_M (this file)** | **8.4 GB** | **~10–11 GB** |

Runs on a 12 GB GPU (RTX 3080/4070), Apple M-series with 16 GB unified memory, or CPU with sufficient RAM.

---

## Training Details

### Base Model
- **Qwen/Qwen3-14B** — 14B parameters, BF16, standard dense transformer (fully compatible with llama.cpp without patches)

### Dataset
- **MedMCQA** — 182,822 training samples, 4,183 validation samples
- 21 medical subjects: Anatomy, Physiology, Biochemistry, Pathology, Pharmacology, Microbiology, Medicine, Surgery, Paediatrics, Gynaecology, Dental, ENT, Ophthalmology, Skin, Psychiatry, Forensic Medicine, Radiology, Orthopaedics, Anaesthesia, and more

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 16 |
| Alpha (α) | 32 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Trainable parameters | 20,971,520 (0.142% of 14B) |

### Training Hyperparameters

| Hyperparameter | Value |
|---------------|-------|
| Training steps | 1,000 (~8.7% of 1 full epoch) |
| Per-device batch size | 4 |
| Gradient accumulation | 4 (effective batch = 16) |
| Learning rate | 2e-4 (cosine decay) |
| Warmup steps | 100 |
| Max sequence length | 512 tokens |
| Precision | BF16 |

### Hardware
- NVIDIA GB10 Grace Blackwell (DGX Spark), 121 GB unified memory
- Training duration: ~83 minutes (1,000 steps at ~5s/step) + 5 evaluation passes

### Validation Metrics (Dev Set, 4,183 samples)

| Checkpoint | Eval Loss | Token Accuracy |
|-----------|-----------|----------------|
| Step 200 | 0.9825 | 78.96% |
| Step 600 | 0.9746 | 79.02% |
| Step 800 | 0.9681 | 79.14% |
| Step 1000 | 0.9664 | 79.18% |
| **Best (saved)** | **0.9649** | **79.20%** |

Eval loss improved consistently throughout training, indicating good generalisation without overfitting.

### Quantization
- **Method:** Q4_K_M via [llama.cpp](https://github.com/ggerganov/llama.cpp)
- F16 GGUF intermediate → Q4_K_M final (intermediate deleted)
- Quantized on DGX Spark (CPU-only quantization, no CUDA required)

---

## Comparison with 9B Version

| | [9B (Qwen3.5-9B)](https://huggingface.co/jamezoon/qwen3-5-9b-medmcqa-gguf) | **This model (Qwen3-14B)** |
|--|---|---|
| Base params | 9B | **14B** |
| Architecture | Hybrid (GatedDeltaNet) | Standard transformer |
| GGUF size | 5.2 GB | **8.4 GB** |
| Min VRAM | ~6–7 GB | ~10–11 GB |
| Best eval loss | 0.9669 | **0.9649** |
| Best token acc | 78.7% | **79.20%** |

The 14B model achieves slightly better validation metrics. Choose the 9B if memory-constrained.

---

## Prompt Format

This model uses the Qwen3 chat template (ChatML):

```
<|im_start|>system
You are a helpful tutor for pre-med students preparing for medical entrance exams.
<|im_end|>
<|im_start|>user
Question: {question}
Options: A. {opa}  B. {opb}  C. {opc}  D. {opd}
Think step by step. Then respond in the format:
Explanation: ...
Answer: <one of A, B, C, D>
<|im_end|>
<|im_start|>assistant
```

---

## See Also

- LoRA adapter (PEFT, not merged): [jamezoon/qwen3-14b-medmcqa-lora](https://huggingface.co/jamezoon/qwen3-14b-medmcqa-lora)
- 9B version (smaller, faster): [jamezoon/qwen3-5-9b-medmcqa-gguf](https://huggingface.co/jamezoon/qwen3-5-9b-medmcqa-gguf)

---

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

---

## Disclaimer

This model is intended for research and educational purposes only. **Do not use it for real clinical decision-making.** Medical decisions must be made by qualified healthcare professionals.
