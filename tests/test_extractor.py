from corpus2dict.extractor import extract_from_text

def test_extract_noun_phrases():
    text = "The government invested heavily in renewable energy and clean drinking water."
    phrases = extract_from_text(text)
    assert "government" in phrases
    assert "renewable energy" in phrases
    assert "clean drinking water" in phrases
    # Ensure articles are stripped
    assert not any(p.startswith("the ") for p in phrases)
    assert not any(p.startswith("a ") for p in phrases)

def test_code_tokens_rejected():
    text = "You need to update the user_id in the database."
    phrases = extract_from_text(text)
    assert "user_id" not in phrases
    assert "database" in phrases
