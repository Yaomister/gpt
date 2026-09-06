from dataclasses import dataclass

@dataclass
class Config:
    n_embedding: int = 768
    batch_size: int = 4
    vocab_size: int = 50304
    n_head: int = 12
    block_size: int = 1024
    dropout: float = 0.1
    bias: bool = True
    n_layers: int = 12
    use_flash_attention: bool = False

    # inference parameters
    use_cache: bool = True

    # training parameters
    grad_clip: float = 1.0
    learning_rate_warmup_epochs: int = 500
    learning_rate_decay_epochs: int = 4000
    evaluation_epochs: int = 1000
    training_epochs: int = 5000
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 3e-5
    weight_decay:int  = 0.1
    accumulation_steps:int = 100
    fp8: bool = True