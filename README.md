# 61.502 Deep Learning — Project: MedMCQA & MCAT

**Authors**: James Oon · Josiah Lau · Tung
**Course**: SUTD MSTR-DAIE · 61.502 Deep Learning · 2026
**Live demo**: https://dl.mdaie-sutd.fit

Three approaches to multiple-choice medical QA on MedMCQA (182K questions) and
MCAT (1,609 held-out MCAT-style questions), plus a serverless AWS exam platform
that lets students take an exam and see per-model evaluation results side by
side with their own score.

## Repository layout

- `medmcqa/` — Generative LLM approach (Qwen3 / Gemma-3 QLoRA + full SFT) and
  the React + FastAPI exam webapp used by the deployed demo.
- `med/medmcqa/` — Cross-encoder discriminative approach (DeBERTa-v3-large) and
  a confidence-gated hybrid pipeline (classifier answer + LLM explainer).
- `aws/` — CDK stack, Lambdas, and deploy scripts for the live webapp
  (Cognito · API Gateway · Step Functions · DynamoDB · S3 · CloudFront · ACM · Route 53).
- `test sets/` — Python evaluation harness for the 1,609-question MCAT
  benchmark. Raw exam content, images, and result artifacts are intentionally
  excluded from this repository.

Training data (MedMCQA dumps), MCAT exam papers, model checkpoints, images, and
any credentials/env files are deliberately excluded (see `.gitignore`).

## Key commands

Training (generative, from `medmcqa/`):

```bash
python training/data_prep.py --data_dir data --output_dir data/processed
bash scripts/run_lora.sh       # LoRA fine-tune (~25 GB GPU)
bash scripts/run_sft.sh        # Full SFT with DeepSpeed ZeRO-3
python training/evaluate.py --compare --results_dir results/
```

Cross-encoder (from `med/medmcqa/`):

```bash
python train.py --model microsoft/deberta-v3-large \
    --train_file ../medmcqa/data/train.json \
    --dev_file   ../medmcqa/data/dev.json --use_context
python evaluate.py --data ../medmcqa/data/dev.json --per-subject
python pipeline.py --question "..." --opa "..." --opb "..." --opc "..." --opd "..."
```

Webapp (UI dev, no GPU needed):

```bash
cd medmcqa/webapp/backend  && MOCK_MODEL=1 uvicorn main:app --reload --port 8000
cd medmcqa/webapp/frontend && npm run dev   # → http://localhost:5173
```

AWS deploy (from repo root):

```bash
# First-time / recovery (runs preflight orphan cleanup):
bash aws/cdk_deploy.sh

# Iterative updates to a healthy stack (additive changes only):
cd aws/cdk && npx --yes aws-cdk@2 deploy --require-approval never
bash aws/deploy_frontend.sh    # rebuild SPA, s3 sync, CloudFront invalidate
```

## Architecture highlights

- Student submits answers → `submit` Lambda scores locally against the answer
  key and kicks off a Step Functions Map state.
- Map state fans out one `grade` Lambda per question (max concurrency 10).
  Each grade task both generates an explanation and runs every configured
  HuggingFace model as an independent test-taker.
- `aggregate` Lambda rolls the per-question model answers into a per-model
  scoreboard on the submission, which the Results page renders next to the
  student's own score.
- Admins can attach figures to questions (MCAT-style); images are uploaded
  through a dedicated Lambda that writes to an `uploads/` prefix on the same
  S3 bucket CloudFront serves.

## License

Coursework. All third-party datasets and model weights remain under their
original licenses.
