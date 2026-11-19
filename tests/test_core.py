import pytest
from app.core import CharTokenizer

def test_tokenizer_round_trip():
    text="Hero: hello!"; tok=CharTokenizer(text)
    assert tok.decode(tok.encode(text)) == text

def test_unknown_character_rejected():
    with pytest.raises(ValueError): CharTokenizer("abc").encode("z")
