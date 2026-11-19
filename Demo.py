"""Streamlit demo: generate game dialogue with FP32 / dynamic INT8 / static INT8 models.
Run from the project root:  streamlit run demo.py   (needs `train` to have been run first)"""
import html, json, time
import pandas as pd
import streamlit as st
import torch

from app.config import ARTIFACTS_DIR, DATA_DIR
from app.dataset import load_corpus, split_data, get_batch, training_corpus_path
from app.quantization import quantize_dynamic_int8, quantize_static_weight_only
from app.storage import load_checkpoint

st.set_page_config(page_title="Quantized Game Dialogue", page_icon="🎮", layout="wide")
st.title("🎮 Quantized Game Dialogue Simulator")
st.caption("Tiny GPT trained from scratch, compared as FP32 vs dynamic INT8 vs static INT8.")


@st.cache_resource(show_spinner="Loading and quantizing models...")
def load_models():
    model, tok, cfg = load_checkpoint(ARTIFACTS_DIR / "model.pt")
    corpus = load_corpus(training_corpus_path(DATA_DIR, ARTIFACTS_DIR))
    train_data, _ = split_data(torch.tensor(tok.encode(corpus), dtype=torch.long))
    calib = [get_batch(train_data, 8, cfg["block_size"])[0] for _ in range(8)]
    return tok, {"FP32 (original)": model,
                 "Dynamic INT8": quantize_dynamic_int8(model),
                 "Static INT8": quantize_static_weight_only(model, calib)}


if not (ARTIFACTS_DIR / "model.pt").exists():
    st.error("No trained model found. Run `python -m app.cli train` first."); st.stop()

tok, models = load_models()

tab_gen, tab_cmp = st.tabs(["Generate dialogue", "Benchmark comparison"])

with tab_gen:
    left, right = st.columns([1, 2])
    with left:
        choice = st.radio("Model", list(models), index=0)
        compare_all = st.checkbox("Generate with all three side by side", value=False)
        prompt = st.text_input("Prompt", "Hero:")
        n_tokens = st.slider("New characters", 20, 400, 160, 10)
        temp = st.slider("Temperature", 0.2, 1.5, 0.8, 0.05)
        seed = st.number_input("Seed", 0, 9999, 7)
        go = st.button("Generate", type="primary")
    with right:
        if go:
            try: ids = tok.encode(prompt)
            except ValueError:
                st.warning("Prompt has characters outside the vocabulary; using 'Hero:'."); ids = tok.encode("Hero:")
            x = torch.tensor([ids], dtype=torch.long)
            names = list(models) if compare_all else [choice]
            cols = st.columns(len(names))
            for col, name in zip(cols, names):
                torch.manual_seed(int(seed))
                t0 = time.perf_counter()
                out = models[name].generate(x, n_tokens, temp)
                dt = time.perf_counter() - t0
                with col:
                    st.subheader(name)
                    st.markdown("<div style='white-space:pre-wrap;word-break:break-word;font-family:monospace;"
                                "background:rgba(128,128,128,.15);padding:12px;border-radius:8px'>"
                                + html.escape(tok.decode(out[0].tolist())) + "</div>", unsafe_allow_html=True)
                    st.caption(f"{n_tokens / dt:.0f} chars/sec on this CPU")
        else:
            st.info("Pick a model, set a prompt and press Generate.")

with tab_cmp:
    path = ARTIFACTS_DIR / "benchmark_report.json"
    if not path.exists():
        st.warning("Run `python -m app.cli benchmark` to see the comparison.")
    else:
        r = json.loads(path.read_text(encoding="utf-8"))
        names = {"original_fp32": "FP32", "dynamic_int8": "Dynamic INT8", "static_int8": "Static INT8"}
        df = pd.DataFrame([{
            "Model": label,
            "Size (KB)": round(r[k]["serialized_size_bytes"] / 1024, 1),
            "Size reduction %": round(r[k]["size_reduction_percent_vs_original"], 1),
            "Latency (ms)": round(r[k]["latency"]["mean_ms"], 2),
            "Measured speedup": round(r[k]["measured_speedup_vs_original"], 2),
            "Simulated edge speedup": round(r[k]["simulated_speedup_vs_original"], 2),
            "Val loss": round(r[k].get("accuracy", {}).get("val_loss", float("nan")), 3),
        } for k, label in names.items()]).set_index("Model")
        st.dataframe(df, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.subheader("Model size (KB)"); c1.bar_chart(df["Size (KB)"])
        c2.subheader("Speedup vs FP32"); c2.bar_chart(df[["Measured speedup", "Simulated edge speedup"]])
        st.caption(" ".join(r.get("notes", [])))
