import argparse, json
from pathlib import Path
import torch
from .config import CONFIG, DATA_DIR, ARTIFACTS_DIR
from .dataset import prepare_corpus, load_corpus, split_data, get_batch, training_corpus_path
from .core import CharTokenizer
from .train import train_model
from .storage import load_checkpoint, save_json
from .quantization import quantize_dynamic_int8, quantize_static_weight_only, save_quantized_checkpoint
from .benchmark import compare_models, format_table

def _data(tok):
    corpus = load_corpus(training_corpus_path(DATA_DIR, ARTIFACTS_DIR))
    ids = torch.tensor(tok.encode(corpus), dtype=torch.long)
    return split_data(ids)

def _calibration(tok, cfg, n=8):
    train_data, _ = _data(tok)
    return [get_batch(train_data, 8, cfg["block_size"])[0] for _ in range(n)]

def prepare(_):
    text = prepare_corpus(DATA_DIR/"game_dialogue.txt", ARTIFACTS_DIR/"prepared_corpus.txt")
    print(f"Prepared {len(text)} characters.")
def train(_):
    _, _, history = train_model(); print(f"Training complete; {len(history)} evaluations saved.")
def generate(args):
    model, tok, _ = load_checkpoint(ARTIFACTS_DIR/"model.pt", CONFIG.device)
    prompt = args.prompt
    try: ids = tok.encode(prompt)
    except ValueError: print("Unknown prompt characters; using Hero:"); ids = tok.encode("Hero:")
    x = torch.tensor([ids],dtype=torch.long,device=CONFIG.device)
    result = model.generate(x,args.max_new_tokens or CONFIG.max_new_tokens,args.temperature)
    text = tok.decode(result[0].tolist()); print(text)
    ARTIFACTS_DIR.mkdir(exist_ok=True); (ARTIFACTS_DIR/"sample_dialogue.txt").write_text(text+"\n", encoding="utf-8")
def quantize(_):
    model,tok,cfg = load_checkpoint(ARTIFACTS_DIR/"model.pt")
    dynamic = quantize_dynamic_int8(model); static = quantize_static_weight_only(model, _calibration(tok, cfg))
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    save_quantized_checkpoint(ARTIFACTS_DIR/"model_dynamic_int8.pt",dynamic,tok,cfg)
    save_quantized_checkpoint(ARTIFACTS_DIR/"model_static_int8.pt",static,tok,cfg)
    print("Saved dynamic and static INT8 checkpoints.")
def benchmark(_):
    original,tok,cfg = load_checkpoint(ARTIFACTS_DIR/"model.pt")
    # Rebuild quantized variants from the original checkpoint. Quantized state_dicts
    # use backend-specific packed parameter formats and are not loaded as float GPTs.
    dynamic = quantize_dynamic_int8(original)
    static = quantize_static_weight_only(original, _calibration(tok, cfg))
    _, val = _data(tok)
    x = get_batch(val, 1, cfg["block_size"])[0]  # full-length sequence
    report = compare_models(original,dynamic,static,x,CONFIG.benchmark_runs,val,cfg["block_size"])
    # sample dialogue from each variant (same prompt, same seed)
    prompt = torch.tensor([tok.encode("Hero:")],dtype=torch.long); samples = {}
    for n, m in (('original_fp32',original),('dynamic_int8',dynamic),('static_int8',static)):
        torch.manual_seed(7); samples[n] = tok.decode(m.generate(prompt,120,0.8)[0].tolist())
    report['samples'] = samples
    save_json(ARTIFACTS_DIR/"benchmark_report.json",report); print(format_table(report))
def report(_):
    p=ARTIFACTS_DIR/"benchmark_report.json"
    if not p.exists(): raise FileNotFoundError("Run benchmark before report.")
    r=json.loads(p.read_text(encoding="utf-8")); print(format_table(r))
    for n,t in r.get('samples',{}).items(): print(f"\n--- {n} sample ---\n{t}")
def all_steps(args):
    for fn in (prepare,train,quantize,benchmark,report): fn(args)
def main():
    parser=argparse.ArgumentParser(description="Quantized Game Dialogue Simulator")
    sub=parser.add_subparsers(dest="command",required=True)
    for name,fn in (("prepare",prepare),("train",train),("quantize",quantize),("benchmark",benchmark),("report",report),("all",all_steps)):
        p=sub.add_parser(name); p.set_defaults(func=fn)
    p=sub.choices["generate"]=sub.add_parser("generate")
    p.add_argument("--prompt",default="Hero:"); p.add_argument("--max-new-tokens",type=int,default=None); p.add_argument("--temperature",type=float,default=.9); p.set_defaults(func=generate)
    args=parser.parse_args(); args.func(args)
if __name__=="__main__": main()
