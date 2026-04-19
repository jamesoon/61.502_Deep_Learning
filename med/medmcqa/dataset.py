"""
MedMCQA dataset loader.

Supports both CSV and JSON formats (auto-detected from file extension).
Returns: (context?, question, options_list, label) tuples.
"""
import json
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset


class MCQADataset(Dataset):

    def __init__(self, data_path: str, use_context: bool = True):
        path = Path(data_path)
        if path.suffix in (".json", ".jsonl"):
            # Try JSONL first (one JSON object per line), fall back to JSON array
            try:
                self.dataset = pd.read_json(path, lines=True)
            except ValueError:
                with open(path) as f:
                    self.dataset = pd.DataFrame(json.load(f))
        else:
            self.dataset = pd.read_csv(path)
        self.dataset = self.dataset.reset_index(drop=True)
        self.use_context = use_context

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        return_tuple = tuple()

        if self.use_context:
            context = row.get("exp", "")
            if pd.isna(context):
                context = ""
            return_tuple += (str(context),)

        question = row.get("question", "")
        if pd.isna(question):
            question = ""
        question = str(question)

        raw_options = [row.get(k, "") for k in ("opa", "opb", "opc", "opd")]
        options = ["" if pd.isna(opt) else str(opt) for opt in raw_options]

        raw_label = row.get("cop", 0)
        if pd.isna(raw_label):
            label = 0
        else:
            label = int(raw_label)
            # Support both 1-based labels (1..4) and 0-based labels (0..3).
            if 1 <= label <= 4:
                label = label - 1

        return_tuple += (question, options, label)
        return return_tuple
