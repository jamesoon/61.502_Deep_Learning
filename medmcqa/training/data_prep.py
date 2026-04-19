"""
Data preparation for Qwen3-30B-A3B fine-tuning on MedMCQA.
Converts raw JSON records to instruction-following JSONL chat format.

Usage:
    python training/data_prep.py --data_dir data --output_dir data/processed
    python training/data_prep.py --data_dir data --output_dir data/processed --max_train 5000
"""

import json
import random
import argparse
from pathlib import Path
from collections import Counter

# cop field maps 1→A, 2→B, 3→C, 4→D
COP_MAP = {1: "A", 2: "B", 3: "C", 4: "D"}

SYSTEM_PROMPT = (
    "You are a helpful tutor for pre-med students preparing for the MCAT. "
    "You answer multiple-choice questions with step-by-step reasoning."
)


def format_user_prompt(record: dict) -> str:
    """Format a MedMCQA record into the Stage 1 prompt template (text-only)."""
    return (
        f"Question: {record['question']}\n"
        f"Image Description: [not used in Stage 1]\n"
        f"Options:\n"
        f"A. {record['opa']}\n"
        f"B. {record['opb']}\n"
        f"C. {record['opc']}\n"
        f"D. {record['opd']}\n\n"
        f"Think step by step. Then respond in the format:\n"
        f"Explanation: ...\n"
        f"Answer: <one of A, B, C, D>"
    )


def format_assistant_response(record: dict) -> str:
    """Format the expected assistant response."""
    answer_letter = COP_MAP.get(record["cop"], "A")
    explanation = (record.get("exp") or "").strip()
    if not explanation:
        explanation = f"The correct answer is {answer_letter} based on medical knowledge."
    return f"Explanation: {explanation}\nAnswer: {answer_letter}"


def to_chat_sample(record: dict) -> dict:
    """Convert a raw record to a chat-format training sample."""
    return {
        "id": record.get("id", ""),
        "subject_name": record.get("subject_name", "Unknown"),
        "topic_name": record.get("topic_name", ""),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_user_prompt(record)},
            {"role": "assistant", "content": format_assistant_response(record)},
        ],
    }


def load_json_or_jsonl(path: str) -> list:
    """Load a file that is either a JSON array or JSONL."""
    with open(path) as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def save_jsonl(records: list, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):,} records → {path}")


def print_stats(records: list, split: str) -> None:
    subjects = Counter(r.get("subject_name", "Unknown") for r in records)
    print(f"\n{split}: {len(records):,} samples")
    for subj, count in subjects.most_common(10):
        print(f"  {subj:<35} {count:>6,}")
    if len(subjects) > 10:
        print(f"  ... and {len(subjects) - 10} more subjects")


def stratify_train(records: list, max_per_subject: int, seed: int) -> list:
    """
    Balance training data across subjects by capping each subject at
    max_per_subject samples (randomly sampled with fixed seed).

    With max_steps=1000 and effective batch=16, the training window is
    ~16,000 samples.  Without stratification, large subjects (Medicine: 9.8%)
    dominate and small subjects (Skin: 1.0%) get only ~155 samples — a 10x
    imbalance that biases model selection toward high-frequency specialties.

    Stratifying to max_per_subject=800 gives 21 subjects × ≤800 ≈ 16,800
    total samples, so every specialty gets equal representation in the
    1000-step training budget.
    """
    rng = random.Random(seed)
    by_subject: dict = {}
    for r in records:
        subj = r.get("subject_name", "Unknown")
        by_subject.setdefault(subj, []).append(r)

    print(f"\n[Stratify] Capping each of {len(by_subject)} subjects to ≤{max_per_subject} samples:")
    flat = []
    for subj in sorted(by_subject):
        pool = by_subject[subj]
        rng.shuffle(pool)
        selected = pool[:max_per_subject]
        print(f"  {subj:<40} {len(pool):>6,} → {len(selected):>6,}")
        flat.extend(selected)

    rng.shuffle(flat)  # mix subjects so batches aren't subject-contiguous
    print(f"[Stratify] Total after stratification: {len(flat):,}")
    return flat


def main():
    parser = argparse.ArgumentParser(description="Prepare MedMCQA data for fine-tuning")
    parser.add_argument("--data_dir", default="data", help="Directory with train/dev/test JSON")
    parser.add_argument("--output_dir", default="data/processed", help="Output JSONL directory")
    parser.add_argument(
        "--max_train",
        type=int,
        default=None,
        help="Cap training samples (e.g. 5000 for quick runs)",
    )
    parser.add_argument(
        "--stratify",
        action="store_true",
        help=(
            "Balance training data across all 21 subjects by capping each to "
            "--max_per_subject samples. Use with max_steps=1000 to ensure every "
            "specialty gets equal representation in the ~16K-sample training window."
        ),
    )
    parser.add_argument(
        "--max_per_subject",
        type=int,
        default=800,
        help="Max samples per subject when --stratify is used (default 800 → ~16,800 total for 21 subjects)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    required_fields = {"question", "opa", "opb", "opc", "opd", "cop"}

    for split in ["train", "dev", "test"]:
        src = data_dir / f"{split}.json"
        if not src.exists():
            print(f"Warning: {src} not found, skipping.")
            continue

        records = load_json_or_jsonl(str(src))

        # Filter invalid records
        valid = [r for r in records if required_fields.issubset(r.keys())]
        dropped = len(records) - len(valid)
        if dropped:
            print(f"Dropped {dropped} invalid records from {split}")

        if split == "train":
            if args.stratify:
                valid = stratify_train(valid, args.max_per_subject, args.seed)
            elif args.max_train:
                random.shuffle(valid)
                valid = valid[: args.max_train]
            else:
                # Shuffle so batches aren't subject-contiguous (original file
                # ordering may group records by subject)
                random.shuffle(valid)

        samples = [to_chat_sample(r) for r in valid]
        save_jsonl(samples, str(output_dir / f"{split}.jsonl"))
        print_stats(valid, split)


if __name__ == "__main__":
    main()
