from corpus2nouns.reducer import reduce_to_pure_nouns

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
