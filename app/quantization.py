import copy
import torch
from torch import nn
from torch.nn import functional as F


class StaticInt8Linear(nn.Module):
    """Static INT8 Linear: per-output-channel INT8 weights + (optionally) a calibrated,
    fixed INT8 activation scale. Uses simulated (fake-quant) arithmetic so it runs on any CPU."""
    def __init__(self, source, act_absmax=None):
        super().__init__()
        weight = source.weight.detach().cpu()
        scale = (weight.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-8)
        self.register_buffer("qweight", torch.clamp(torch.round(weight / scale), -127, 127).to(torch.int8))
        self.register_buffer("scale", scale.float())
        if source.bias is None: self.register_parameter("bias", None)
        else: self.register_buffer("bias", source.bias.detach().cpu().clone())
        act_scale = None if act_absmax is None else max(float(act_absmax) / 127.0, 1e-8)
        self.register_buffer("act_scale", None if act_scale is None else torch.tensor(act_scale))
        self.in_features, self.out_features = source.in_features, source.out_features

    @property
    def weight(self):
        return self.qweight.float() * self.scale

    def forward(self, x):
        if self.act_scale is not None:  # static activation quantization (scale fixed at calibration time)
            x = torch.clamp(torch.round(x / self.act_scale), -127, 127) * self.act_scale
        return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


@torch.no_grad()
def calibrate(model, calibration_batches):
    """Run sample inputs through the FP32 model and record max |activation| entering each Linear."""
    stats, hooks = {}, []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_pre_hook(
                lambda mod, args, n=name: stats.__setitem__(n, max(stats.get(n, 0.0), float(args[0].abs().max())))))
    model.eval()
    for batch in calibration_batches: model(batch)
    for h in hooks: h.remove()
    return stats


def _replace(module, stats, prefix=""):
    for name, child in list(module.named_children()):
        full = f"{prefix}{name}"
        if isinstance(child, nn.Linear):
            setattr(module, name, StaticInt8Linear(child, stats.get(full)))
        else:
            _replace(child, stats, full + ".")
    return module


def quantize_static_weight_only(model, calibration_batches=None):
    """Static INT8 PTQ of every Linear layer. With calibration_batches, activations are also
    quantized using fixed calibrated scales (full static PTQ); without, weight-only."""
    base = copy.deepcopy(model).cpu().eval()
    stats = calibrate(base, calibration_batches) if calibration_batches else {}
    return _replace(base, stats)


def quantize_dynamic_int8(model):
    """PyTorch dynamic PTQ: INT8 weights, activations quantized on the fly at inference."""
    try:
        return torch.ao.quantization.quantize_dynamic(
            copy.deepcopy(model).cpu().eval(), {nn.Linear}, dtype=torch.qint8)
    except (AttributeError, RuntimeError, NotImplementedError) as exc:
        raise RuntimeError("Dynamic quantization unavailable; install a CPU-enabled PyTorch build.") from exc


def save_quantized_checkpoint(path, model, tokenizer, model_config):
    torch.save({"model_state": model.state_dict(), "tokenizer": tokenizer.to_dict(), "model_config": model_config}, path)
