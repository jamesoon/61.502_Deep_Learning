"""
Cross-encoder MCQ classifier.

Architecture change from the original per-option encoder:
  OLD:  encode([Q + opt_i]) x 4  ->  linear(CLS) -> [4] logits
  NEW:  encode([Q | A. ... B. ... C. ... D. ...])  ->  MLP(CLS) -> [4] logits

The cross-encoder sees all options in one forward pass, enabling the model
to attend across options (spot distractors, compare relative plausibility).

Supports: DeBERTa-v3-large (recommended), BioLinkBERT-large, PubMedBERT-large,
or any HF AutoModel with a pooler / CLS output.
"""

import functools

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModel, AutoTokenizer


class MCQAModel(nn.Module):
    """Cross-encoder: single-sequence classification over all 4 options."""

    def __init__(self, model_name_or_path: str, args: dict):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.args = args

        hidden_size = self.model.config.hidden_size
        self.args["hidden_size"] = hidden_size

        num_choices = args.get("num_choices", 4)
        mlp_hidden = args.get("mlp_hidden", 256)
        dropout_rate = args.get("hidden_dropout_prob", 0.1)

        # 2-layer MLP head: hidden -> mlp_hidden -> num_choices
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden, num_choices),
        )

        # Label smoothing CE loss
        label_smoothing = args.get("label_smoothing", 0.1)
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # ── Speed optimizations ───────────────────────────────────────────
        # Freeze bottom encoder layers (keep top N trainable)
        freeze_layers = args.get("freeze_layers", 0)
        if freeze_layers > 0:
            self._freeze_bottom_layers(freeze_layers)

        # Gradient checkpointing (trades compute for memory → bigger batch)
        if args.get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()

    def _freeze_bottom_layers(self, n_freeze: int):
        """Freeze embeddings + bottom n_freeze transformer layers."""
        # Freeze embeddings
        if hasattr(self.model, "embeddings"):
            for p in self.model.embeddings.parameters():
                p.requires_grad = False

        # Freeze encoder layers
        encoder = None
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
            encoder = self.model.encoder.layer
        elif hasattr(self.model, "layer"):
            encoder = self.model.layer

        if encoder is not None:
            n_freeze = min(n_freeze, len(encoder))
            for layer in encoder[:n_freeze]:
                for p in layer.parameters():
                    p.requires_grad = False
            total = len(encoder)
            print(f"[Model] Froze embeddings + {n_freeze}/{total} encoder layers "
                  f"(training top {total - n_freeze} layers + head)")

    # ── Forward ───────────────────────────────────────────────────────────

    def _get_cls_embedding(self, input_ids, attention_mask, **kwargs):
        """Extract CLS embedding, handling models with/without pooler."""
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in kwargs and kwargs["token_type_ids"] is not None:
            try:
                outputs = self.model(**model_inputs, token_type_ids=kwargs["token_type_ids"])
                return outputs[1] if len(outputs) > 1 and outputs[1] is not None else outputs[0][:, 0]
            except TypeError:
                pass
        outputs = self.model(**model_inputs)
        if len(outputs) > 1 and outputs[1] is not None:
            return outputs[1]
        return outputs[0][:, 0]

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        cls_emb = self._get_cls_embedding(
            input_ids, attention_mask, token_type_ids=token_type_ids
        )
        logits = self.head(cls_emb)  # [batch, num_choices]
        return logits

    # ── Collate: cross-encoder format ─────────────────────────────────────

    def process_batch(self, batch, tokenizer, max_len=256):
        """
        Cross-encoder collation: concatenate question + all options into one
        sequence per sample.  Uses dynamic padding (pad to longest in batch,
        capped at max_len) instead of always padding to max_len.
        """
        texts = []
        labels = []
        for data_tuple in batch:
            if len(data_tuple) == 4:
                context, question, options, label = data_tuple
            else:
                context = None
                question, options, label = data_tuple

            option_text = " ".join(
                f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
            )
            text = f"Question: {question} Options: {option_text}"
            if context and str(context).strip():
                text = f"Context: {context} {text}"

            texts.append(text)
            labels.append(label)

        # Dynamic padding: pad to longest sequence in batch, not max_len
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="longest",
            max_length=max_len,
            return_tensors="pt",
        )
        return tokenized, torch.tensor(labels)

    # ── DataLoader helpers ────────────────────────────────────────────────

    def get_train_dataloader(self, dataset):
        collate = functools.partial(
            self.process_batch,
            tokenizer=self.tokenizer,
            max_len=self.args["max_len"],
        )
        return DataLoader(
            dataset,
            batch_size=self.args["batch_size"],
            sampler=RandomSampler(dataset),
            collate_fn=collate,
            num_workers=2,
            pin_memory=True,
        )

    def get_val_dataloader(self, dataset):
        collate = functools.partial(
            self.process_batch,
            tokenizer=self.tokenizer,
            max_len=self.args["max_len"],
        )
        return DataLoader(
            dataset,
            batch_size=self.args["batch_size"] * 2,  # can use bigger batch for eval
            sampler=SequentialSampler(dataset),
            collate_fn=collate,
            num_workers=2,
            pin_memory=True,
        )

    # ── Checkpoint helpers ────────────────────────────────────────────────

    def save_checkpoint(self, path: str, model_name: str, epoch: int,
                        val_loss: float, val_acc: float):
        torch.save(
            {
                "model_name": model_name,
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "args": self.args,
                "model_state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> "MCQAModel":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(ckpt["model_name"], ckpt["args"])
        model.load_state_dict(ckpt["model_state_dict"])
        return model
