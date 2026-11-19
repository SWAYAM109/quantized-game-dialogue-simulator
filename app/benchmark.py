import io, math, time
import torch
from torch import nn
from .quantization import StaticInt8Linear

QUANT_LINEAR_NAMES = ("DynamicQuantizedLinear",)


def _is_linear(m):
    if isinstance(m, (nn.Linear, StaticInt8Linear)): return True
    # torch.ao dynamic-quantized Linear lives in a "quantized.dynamic" module and is named just "Linear"
    return "quantized" in type(m).__module__ and hasattr(m, "in_features") and hasattr(m, "out_features")


def _is_fp32(m):
    return isinstance(m, nn.Linear) and not isinstance(m, StaticInt8Linear)


def _dims(m):
    return (m.in_features, m.out_features)


def serialized_size_bytes(model):
    b = io.BytesIO(); torch.save(model.state_dict(), b); return b.tell()


def count_linear_macs(model, sequence_length):
    """Multiply-accumulates in all Linear layers (counts dynamic-quantized layers too)."""
    return int(sum(_dims(m)[0] * _dims(m)[1] * sequence_length for m in model.modules() if _is_linear(m)))


def weight_bytes_read(model, bytes_per_weight_by_kind=None):
    """Simulated memory traffic per forward pass: bytes of Linear weights read.
    FP32 = 4 B/weight, INT8 = 1 B/weight. Small models on edge devices are memory-bound."""
    total = 0
    for m in model.modules():
        if _is_linear(m):
            i, o = _dims(m)
            total += i * o * (4 if _is_fp32(m) else 1)
    return total


def simulated_cost(model, seq_len, int8_op_cost=0.25):
    """Toy edge-device cost model: an INT8 MAC costs `int8_op_cost` of an FP32 MAC (typical ~2-4x INT8 speedup)."""
    cost = 0.0
    for m in model.modules():
        if _is_linear(m):
            i, o = _dims(m)
            cost += i * o * seq_len * (1.0 if _is_fp32(m) else int8_op_cost)
    return cost


@torch.no_grad()
def measure_latency(model, input_ids, runs=20, warmup=3):
    model.eval()
    for _ in range(warmup): model(input_ids)
    times = []
    for _ in range(runs):
        start = time.perf_counter(); model(input_ids)
        times.append((time.perf_counter() - start) * 1000)
    return {"mean_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times), "runs": runs}


@torch.no_grad()
def evaluate_loss(model, data, block_size, batches=20, batch_size=16):
    """Validation loss + perplexity, using a fixed seed so every variant sees the same batches."""
    from .dataset import get_batch
    model.eval(); g = torch.get_rng_state(); torch.manual_seed(1234)
    losses = [float(model(*get_batch(data, batch_size, block_size))[1]) for _ in range(batches)]
    torch.set_rng_state(g)
    loss = sum(losses) / len(losses)
    return {"val_loss": loss, "perplexity": math.exp(loss)}


def compare_models(original, dynamic, static, input_ids, runs=20, val_data=None, block_size=None):
    models = {"original_fp32": original, "dynamic_int8": dynamic, "static_int8": static}
    seq = input_ids.shape[1]
    report = {}
    for name, model in models.items():
        row = {"serialized_size_bytes": serialized_size_bytes(model),
               "linear_macs": count_linear_macs(model, seq),
               "weight_bytes_read": weight_bytes_read(model),
               "simulated_relative_cost": simulated_cost(model, seq),
               "latency": measure_latency(model, input_ids, runs)}
        if val_data is not None:
            row["accuracy"] = evaluate_loss(model, val_data, block_size or seq)
        report[name] = row
    base = report["original_fp32"]
    for row in report.values():
        row["size_reduction_percent_vs_original"] = 100 * (1 - row["serialized_size_bytes"] / base["serialized_size_bytes"])
        row["simulated_speedup_vs_original"] = base["simulated_relative_cost"] / max(row["simulated_relative_cost"], 1e-9)
        row["measured_speedup_vs_original"] = base["latency"]["mean_ms"] / max(row["latency"]["mean_ms"], 1e-9)
    report["notes"] = [
        "Measured latency depends on hardware and load; static INT8 here uses simulated (fake-quant) arithmetic, so it may be slower than FP32 on a laptop CPU.",
        "simulated_speedup assumes an edge device with native INT8 kernels where an INT8 MAC costs 0.25x an FP32 MAC.",
        "weight_bytes_read models memory traffic: FP32=4 bytes/weight, INT8=1 byte/weight.",
    ]
    return report


def format_table(report):
    if "measured_speedup_vs_original" not in report.get("original_fp32", {}):
        raise ValueError("benchmark_report.json is from an older version. Re-run: python -m app.cli benchmark")
    cols = [("Model", None), ("Size KB", "serialized_size_bytes"), ("Size -%", "size_reduction_percent_vs_original"),
            ("Latency ms", "latency"), ("Meas. x", "measured_speedup_vs_original"),
            ("Sim. x", "simulated_speedup_vs_original"), ("Val loss", "accuracy"), ("PPL", "accuracy")]
    lines = [" | ".join(c for c, _ in cols), "-" * 78]
    for name in ("original_fp32", "dynamic_int8", "static_int8"):
        r = report[name]; acc = r.get("accuracy", {})
        lines.append(" | ".join([name, f"{r['serialized_size_bytes']/1024:.1f}", f"{r['size_reduction_percent_vs_original']:.1f}",
                                 f"{r['latency']['mean_ms']:.2f}", f"{r['measured_speedup_vs_original']:.2f}",
                                 f"{r['simulated_speedup_vs_original']:.2f}",
                                 f"{acc.get('val_loss', float('nan')):.3f}", f"{acc.get('perplexity', float('nan')):.2f}"]))
    return "\n".join(lines) + "\n\n" + "\n".join("* " + n for n in report["notes"])
