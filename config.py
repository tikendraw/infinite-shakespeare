from dataclasses import dataclass
import torch


# config
@dataclass
class GPTConfig:
    block_size: int = 500  # context_length
    vocab_size: int = 1355
    batch_size: int = 32
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.05
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rate: int = 1e-2
    epochs: int = 100
    reserved_token: int = 0