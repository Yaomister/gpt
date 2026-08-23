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