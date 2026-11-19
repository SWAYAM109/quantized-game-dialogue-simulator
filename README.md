# Quantized Game Dialogue Simulator

A complete educational project that trains a small decoder-only GPT from scratch on a custom game dialogue corpus, generates dialogue, applies dynamic and static INT8 post-training quantization, and compares serialized model size and simulated inference performance.

## Requirements
- Python 3.12 (recommended, as specified in the project brief)
- Git
- CPU is sufficient; GPU is optional
- PyTorch 2.2+

## Setup — Windows PowerShell
```powershell
cd quantized-game-dialogue-simulator
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```
If activation is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again.

## Setup — macOS / Linux
```bash
cd quantized-game-dialogue-simulator
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

## Run the end-to-end pipeline
Run these commands from the repository root, with the virtual environment activated:
```bash
python -m app.cli prepare
python -m app.cli train
python -m app.cli generate --prompt "Hero:"
python -m app.cli quantize
python -m app.cli benchmark
python -m app.cli report
```
Or run `python -m app.cli all` to execute the complete pipeline in order. The first training run uses a small model and 250 steps by default; increase `TRAIN_STEPS` in `.env` after validating the setup.

## What is implemented
1. **Corpus preparation:** validates and normalizes `data/game_dialogue.txt`.
2. **GPT from scratch:** character tokenizer, token/position embeddings, causal self-attention Transformer blocks, and next-character prediction.
3. **Training:** AdamW optimization, gradient clipping, train/validation loss snapshots, and checkpoint saving.
4. **Generation:** autoregressive sampling from a user prompt.
5. **Dynamic INT8 PTQ:** PyTorch dynamic quantization of eligible Linear modules.
6. **Static INT8 PTQ:** per-layer symmetric weight-only quantization. INT8 weights and scales are stored; the portable reference forward pass dequantizes weights before `Linear`. This demonstrates static quantized storage and its accuracy/size trade-off, but is not an optimized integer inference kernel.
7. **Comparison:** checkpoint state size, measured wall-clock latency, and approximate Linear-layer multiply-accumulate counts.
8. **CI:** GitHub Actions installs Python 3.12 dependencies and runs pytest.

Latency depends on hardware and system load. Quantization does not guarantee faster execution on every machine. The static reference path may be slower because it dequantizes weights at runtime; the benchmark reports measured results rather than assuming a speedup.

## CLI commands
- `prepare` — prepare corpus
- `train` — train and save model
- `generate --prompt "Hero:"` — generate a dialogue continuation
- `quantize` — save dynamic and static INT8 variants
- `benchmark` — compare all three variants
- `report` — print saved benchmark report
- `all` — run all stages

## Repository layout
```text
quantized-game-dialogue-simulator/
├── .env.example
├── .gitignore
├── .github/workflows/ci.yml
├── README.md
├── requirements.txt
├── pytest.ini
├── data/game_dialogue.txt
├── app/
│   ├── __init__.py
│   ├── benchmark.py
│   ├── cli.py
│   ├── config.py
│   ├── core.py
│   ├── dataset.py
│   ├── model.py
│   ├── quantization.py
│   ├── storage.py
│   └── train.py
├── scripts/verify_environment.py
├── scripts/run_all.py
├── tests/
│   ├── test_core.py
│   ├── test_model.py
│   ├── test_quantization.py
│   └── test_smoke.py
└── artifacts/ (generated files; ignored by Git)
```

## Tests
```bash
pytest -q
```
Tests cover tokenizer behavior, causal model output and loss, sequence-length validation, generation, batching, backward propagation, and static quantization. The tests are intentionally small and do not run the full default training job.

## Notes
- No API key, cloud account, or hosting service is required. `.env.example` includes optional future integration placeholders only.
- The included corpus is a small original fantasy-adventure sample. Add more legally usable dialogue to improve training quality.
- This is a small learning model, not a pretrained foundation model or production dialogue system.


## Updates (v2)
- **GPT is now fully from scratch**: custom `CausalSelfAttention` + `Block` built from `nn.Linear` (no `nn.TransformerEncoder`).
- **Corpus** expanded with `python scripts/generate_corpus.py` (~30 KB); training uses the cleaned `artifacts/prepared_corpus.txt` (run `prepare` first).
- **Dynamic INT8** now quantizes every Linear layer (attention QKV/proj, feed-forward, `lm_head`).
- **Static INT8** is per-channel INT8 weights + calibrated fixed activation scales (`calibrate()`).
- **Benchmark** reports size, measured latency, simulated edge-device speedup (INT8 MAC = 0.25x FP32), weight bytes read, validation loss/perplexity, and sample dialogue per variant. `python -m app.cli report` prints a comparison table.
