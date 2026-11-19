from pathlib import Path
import torch
from app.core import CharTokenizer
from app.dataset import load_corpus, split_data, get_batch
from app.model import GPTLanguageModel
from app.quantization import quantize_static_weight_only

def test_corpus_batch_smoke():
    text=load_corpus(Path(__file__).resolve().parents[1]/"data"/"game_dialogue.txt")
    ids=torch.tensor(CharTokenizer(text).encode(text)); train,val=split_data(ids)
    x,y=get_batch(train,2,8)
    assert x.shape==y.shape==(2,8) and len(val)>0

def test_backward_and_quantization_smoke():
    model=GPTLanguageModel(15,8,16,4,1,0); x=torch.randint(0,15,(2,8))
    _,loss=model(x,x); loss.backward()
    assert quantize_static_weight_only(model)(x)[0].shape==(2,8,15)
