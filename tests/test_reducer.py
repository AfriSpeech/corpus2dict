from corpus2dict.reducer import reduce_to_pure_nouns

def test_reduce_adjectives():
    pairs = [
        ("renewable energy", 100),
        ("clean energy", 50),
        ("energy", 500),
        ("modern corporate office", 20),
        ("office", 80),
        ("space station", 40),
        ("international space station", 30)
    ]
    reduced = reduce_to_pure_nouns(pairs)
    assert reduced["energy"] == 650
    assert reduced["office"] == 100
    assert reduced["space station"] == 70

def test_classify_noun_type():
    from corpus2dict.reducer import classify_noun_type
    mock_type_map = {
        "fda": "proper",
        "hiv": "proper",
        "paris": "proper",
        "united": "proper",
        "states": "common",
        "water": "common",
        "students": "common",
        "algorithm": "common"
    }
    assert classify_noun_type("fda", mock_type_map) == "proper"
    assert classify_noun_type("hiv", mock_type_map) == "proper"
    assert classify_noun_type("united states", mock_type_map) == "proper"
    assert classify_noun_type("water", mock_type_map) == "common"
    assert classify_noun_type("students", mock_type_map) == "common"
    assert classify_noun_type("algorithm", mock_type_map) == "common"
