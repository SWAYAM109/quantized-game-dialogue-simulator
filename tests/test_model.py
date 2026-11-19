import torch
from app.model import GPTLanguageModel

def tiny(): return GPTLanguageModel(12,8,16,4,1,0)
def test_forward_shape_and_loss():
    model=tiny(); x=torch.randint(0,12,(2,8)); logits,loss=model(x,x)
    assert logits.shape==(2,8,12) and torch.isfinite(loss)
def test_long_sequence_rejected():
    try: tiny()(torch.zeros((1,9),dtype=torch.long))
    except ValueError: return
    assert False, "Expected block size validation"
def test_generation_length():
    assert tiny().generate(torch.zeros((1,3),dtype=torch.long),2).shape==(1,5)

def test_causality():
    m=tiny().eval(); a=torch.randint(0,12,(1,8)); b=a.clone(); b[0,-1]=(b[0,-1]+1)%12
    assert torch.allclose(m(a)[0][:,:-1],m(b)[0][:,:-1],atol=1e-5)
