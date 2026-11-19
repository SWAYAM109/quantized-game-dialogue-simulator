import json
from pathlib import Path
import torch
from .config import ARTIFACTS_DIR

def ensure_artifacts(): ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def save_checkpoint(path, model, tokenizer, model_config):
    ensure_artifacts()
    torch.save({"model_state": model.state_dict(), "tokenizer": tokenizer.to_dict(), "model_config": model_config}, path)

def load_checkpoint(path, device="cpu"):
    path = Path(path)
    if not path.exists(): raise FileNotFoundError(f"Checkpoint not found: {path}. Run `python -m app.cli train` first.")
    try: payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError: payload = torch.load(path, map_location=device)
    from .core import CharTokenizer
    from .model import GPTLanguageModel
    cfg = payload["model_config"]
    tokenizer = CharTokenizer.from_dict(payload["tokenizer"])
    model = GPTLanguageModel(**cfg)
    model.load_state_dict(payload["model_state"])
    return model.to(device).eval(), tokenizer, cfg

def save_json(path, payload):
    ensure_artifacts()
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
