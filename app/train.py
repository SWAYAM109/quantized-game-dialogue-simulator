from pathlib import Path
import torch
from .config import CONFIG, DATA_DIR, ARTIFACTS_DIR
from .core import CharTokenizer, set_seed
from .dataset import load_corpus, split_data, get_batch, training_corpus_path
from .model import GPTLanguageModel
from .storage import save_checkpoint, save_json

def estimate_loss(model, train_data, val_data, batch_size, block_size, device, eval_iters=10):
    model.eval(); result = {}
    with torch.no_grad():
        for name, data in (("train", train_data), ("validation", val_data)):
            losses = []
            for _ in range(eval_iters):
                x, y = get_batch(data, batch_size, block_size, device)
                _, loss = model(x, y); losses.append(float(loss.item()))
            result[name] = sum(losses) / len(losses)
    model.train(); return result

def train_model(steps=None, save_path=None, overrides=None):
    c = CONFIG
    settings = {"block_size":c.block_size,"n_embd":c.n_embd,"n_head":c.n_head,"n_layer":c.n_layer,"dropout":c.dropout}
    if overrides: settings.update(overrides)
    steps = c.train_steps if steps is None else steps
    device = c.device if c.device == "cpu" or torch.cuda.is_available() else "cpu"
    set_seed(c.seed)
    corpus = load_corpus(training_corpus_path(DATA_DIR, ARTIFACTS_DIR))
    tokenizer = CharTokenizer(corpus)
    encoded = torch.tensor(tokenizer.encode(corpus), dtype=torch.long)
    train_data, val_data = split_data(encoded)
    model = GPTLanguageModel(tokenizer.vocab_size, **settings).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.learning_rate)
    history = []
    for step in range(steps):
        if step % c.eval_interval == 0 or step == steps - 1:
            metrics = estimate_loss(model, train_data, val_data, c.batch_size, settings["block_size"], device)
            metrics["step"] = step; history.append(metrics)
        x, y = get_batch(train_data, c.batch_size, settings["block_size"], device)
        _, loss = model(x, y); optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    save_path = Path(save_path or ARTIFACTS_DIR / "model.pt")
    save_checkpoint(save_path, model, tokenizer, {"vocab_size":tokenizer.vocab_size, **settings})
    save_json(ARTIFACTS_DIR / "training_history.json", {"history":history,"steps":steps})
    return model, tokenizer, history
