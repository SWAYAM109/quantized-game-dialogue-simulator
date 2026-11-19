import torch
from app.model import GPTLanguageModel
from app.quantization import quantize_static_weight_only, StaticInt8Linear

def test_static_quantization_runs_and_stores_int8():
    model=GPTLanguageModel(10,8,16,4,1,0).eval()
    quantized=quantize_static_weight_only(model)
    linear=next(m for m in quantized.modules() if isinstance(m,StaticInt8Linear))
    assert linear.qweight.dtype==torch.int8
    out,_=quantized(torch.randint(0,10,(1,5)))
    assert out.shape==(1,5,10)

def test_dynamic_quantization_output_shape():
    from app.quantization import quantize_dynamic_int8
    model=GPTLanguageModel(10,8,16,4,1,0).eval()
    quantized=quantize_dynamic_int8(model)
    out,_=quantized(torch.randint(0,10,(1,5)))
    assert out.shape==(1,5,10)

def test_static_with_activation_calibration_and_all_linears_quantized():
    from torch import nn
    from app.quantization import quantize_static_weight_only
    model=GPTLanguageModel(10,8,16,4,1,0).eval()
    x=torch.randint(0,10,(2,8))
    q=quantize_static_weight_only(model,[x])
    assert not any(type(m) is nn.Linear for m in q.modules())
    assert next(m for m in q.modules() if isinstance(m,StaticInt8Linear)).act_scale is not None
    assert q(x)[0].shape==(2,8,10)
