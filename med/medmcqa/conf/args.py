from dataclasses import dataclass, field


@dataclass
class Arguments:
    train_csv: str
    test_csv: str
    dev_csv: str

    # Training
    batch_size: int = 8           # micro-batch (bigger now thanks to freeze + grad ckpt)
    grad_accum_steps: int = 2     # effective batch = 8 * 2 = 16
    max_len: int = 256            # MedMCQA Q+4opts rarely exceeds 200 tokens
    learning_rate: float = 2e-5   # lower LR for large pre-trained encoders
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06    # 6% of steps for linear warmup
    num_epochs: int = 5
    patience: int = 3             # early stopping patience

    # Model
    pretrained_model_name: str = "microsoft/deberta-v3-large"
    hidden_size: int = 1024       # auto-overridden at runtime
    hidden_dropout_prob: float = 0.1
    mlp_hidden: int = 256         # hidden dim of 2-layer MLP head
    num_choices: int = 4
    label_smoothing: float = 0.1

    # Speed optimizations
    freeze_layers: int = 18       # freeze bottom 18/24 layers (train top 6 + head)
    gradient_checkpointing: bool = True

    # Hardware
    device: str = "cuda"
    gpu: str = "0"

    # Data
    use_context: bool = True

    # Legacy compat
    checkpoint_batch_size: int = 32
    print_freq: int = 100
