import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        if not chars:
            raise ValueError("Tokenizer cannot be built from empty text.")
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list[int]:
        unknown = sorted(set(text) - set(self.stoi))
        if unknown:
            raise ValueError(f"Text contains characters outside vocabulary: {unknown!r}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def to_dict(self):
        return {"chars": list(self.stoi.keys())}

    @classmethod
    def from_dict(cls, data):
        return cls("".join(data["chars"]))
