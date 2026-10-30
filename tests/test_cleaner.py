import pytest
from corpus2nouns.cleaner import strip_structured_text, is_code_token, chunk_long_text

def test_strip_code_blocks():
    text = "Here is an explanation:\n```python\ndef foo():\n    return 42\n```\nIt works well."
    cleaned = strip_structured_text(text)
    assert "def foo" not in cleaned
    assert "explanation" in cleaned
    assert "works well" in cleaned

def test_strip_inline_code():
    text = "Please update `user_id` and restart `server.js`."
    cleaned = strip_structured_text(text)
    assert "user_id" not in cleaned
    assert "server.js" not in cleaned
    assert "Please update and restart" in cleaned

def test_strip_math():
    text = "The formula is $f(x) = x^2$ and $$\\int_0^1 x dx = 1/2$$."
    cleaned = strip_structured_text(text)
    assert "x^2" not in cleaned
    assert "The formula is and" in cleaned

def test_is_code_token():
    assert is_code_token("user_id") is True
    assert is_code_token("api_key") is True
    assert is_code_token("fooBar") is True
    assert is_code_token("MyClass") is True
    assert is_code_token("script.py") is True
    assert is_code_token("0x123") is True
    assert is_code_token("renewable") is False
    assert is_code_token("energy") is False
    assert is_code_token("spacecraft") is False

def test_chunk_long_text():
    text = "short text"
    assert chunk_long_text(text, 50) == ["short text"]
    long_text = "word " * 100
    chunks = chunk_long_text(long_text, 100)
    assert len(chunks) > 1
