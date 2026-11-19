from dataclasses import dataclass
from pathlib import Path
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
@dataclass(frozen=True)
class Config:
    seed: int = int(os.getenv("SEED", "42"))
    device: str = os.getenv("DEVICE", "cpu")
    block_size: int = int(os.getenv("BLOCK_SIZE", "96"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "16"))
    n_embd: int = int(os.getenv("N_EMBD", "96"))
    n_head: int = int(os.getenv("N_HEAD", "4"))
    n_layer: int = int(os.getenv("N_LAYER", "2"))
    dropout: float = float(os.getenv("DROPOUT", "0.1"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "0.001"))
    train_steps: int = int(os.getenv("TRAIN_STEPS", "1500"))
    eval_interval: int = int(os.getenv("EVAL_INTERVAL", "250"))
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "240"))
    benchmark_runs: int = int(os.getenv("BENCHMARK_RUNS", "30"))
CONFIG = Config()
