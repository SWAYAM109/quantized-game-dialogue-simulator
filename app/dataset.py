from pathlib import Path
import torch

def load_corpus(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if len(text) < 20:
        raise ValueError("Corpus is too small; provide at least 20 characters.")
    return text

def prepare_corpus(source: Path, destination: Path) -> str:
    text = load_corpus(source)
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(normalized + "\n", encoding="utf-8")
    return normalized

def split_data(encoded: torch.Tensor, train_fraction: float = 0.9):
    if encoded.numel() < 4:
        raise ValueError("Encoded corpus must contain at least four tokens.")
    split = max(2, min(int(len(encoded) * train_fraction), len(encoded) - 2))
    return encoded[:split], encoded[split:]

def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device="cpu"):
    if len(data) <= block_size:
        raise ValueError(f"Need more than block_size={block_size} tokens; got {len(data)}.")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

def training_corpus_path(data_dir: Path, artifacts_dir: Path) -> Path:
    """Use the cleaned corpus produced by `prepare` when it exists, else the raw file."""
    prepared = artifacts_dir / "prepared_corpus.txt"
    return prepared if prepared.exists() else data_dir / "game_dialogue.txt"
