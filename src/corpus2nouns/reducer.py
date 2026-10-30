"""Reducer module for stripping adjectives, determiners, and adverbs to produce pure nouns."""

from collections import Counter
import spacy

_reducer_nlp = None

def get_reducer_nlp():
    global _reducer_nlp
    if _reducer_nlp is None:
        _reducer_nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
    return _reducer_nlp

def reduce_phrase_to_pure_noun(phrase: str, doc, stopwords: set) -> str | None:
    """Drop adjectives, determiners, and adverbs from a single phrase Doc."""
    kept = []
    for t in doc:
        # Drop adjectives, determiners, adverbs, and stopwords
        if t.pos_ in ("ADJ", "DET", "ADV") or t.is_stop or t.text.lower() in stopwords:
            continue
        # Retain nominal head tokens (NOUN, PROPN)
        if t.pos_ in ("NOUN", "PROPN"):
            kept.append(t.text.lower())
            
    if kept and len(kept) <= 5:
        if not any(len(w) < 2 and w != "ai" for w in kept):
            return " ".join(kept)
    return None

def reduce_to_pure_nouns(phrase_freq_pairs: list[tuple[str, int]], nlp=None, batch_size: int = 2000) -> Counter:
    """Take a list of (phrase, frequency) pairs, strip adjectives/determiners, and deduplicate with summed counts."""
    if nlp is None:
        nlp = get_reducer_nlp()
        
    stopwords = nlp.Defaults.stop_words | {
        "a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their",
        "all", "some", "any", "each", "every", "other", "another", "such", "no", "what", "which", "whose",
        "few", "many", "much", "more", "most", "several", "both", "either", "neither", "own"
    }
    
    phrases = [p for p, _ in phrase_freq_pairs]
    freqs = [f for _, f in phrase_freq_pairs]
    
    deduped_counter = Counter()
    for phrase, freq, doc in zip(phrases, freqs, nlp.pipe(phrases, batch_size=batch_size)):
        pure_noun = reduce_phrase_to_pure_noun(phrase, doc, stopwords)
        if pure_noun:
            deduped_counter[pure_noun] += freq
            
    return deduped_counter
