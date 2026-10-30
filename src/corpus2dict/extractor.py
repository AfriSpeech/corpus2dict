"""Extractor module using spaCy noun chunking with article/stopword stripping."""

import re
import spacy
from corpus2dict.cleaner import is_code_token, CODE_KEYWORDS

# Common stopwords and articles to strip from boundaries of chunks
STOPWORDS_BASE = {
    "a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their",
    "all", "some", "any", "each", "every", "other", "another", "such", "no", "what", "which", "whose",
    "few", "many", "much", "more", "most", "several", "both", "either", "neither", "own"
}

_nlp = None

def get_nlp():
    """Lazy-load spaCy English model with unnecessary components disabled."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        _nlp.max_length = 5000000
    return _nlp

def extract_noun_phrases(doc) -> list[str]:
    """Extract clean noun phrases from a spaCy Doc.
    
    - Requires syntactic root to be NOUN or PROPN.
    - Strips leading and trailing determiners, articles, pronouns, and stopwords.
    - Discards chunks containing code tokens, snake_case, camelCase, or code symbols.
    - Normalizes to lowercase and enforces word count constraints (1-6 words).
    """
    stopwords = get_nlp().Defaults.stop_words | STOPWORDS_BASE
    phrases = []
    
    for chunk in doc.noun_chunks:
        # 1. Syntactic root check
        if chunk.root.pos_ not in ("NOUN", "PROPN"):
            continue
            
        # 2. Extract tokens, excluding whitespace and punctuation
        tokens = [t for t in chunk if not t.is_space and not t.is_punct]
        if not tokens:
            continue
            
        # 3. Strip leading articles, determiners, pronouns, and stopwords
        start_idx = 0
        while start_idx < len(tokens) and (
            tokens[start_idx].text.lower() in stopwords
            or tokens[start_idx].pos_ in ("DET", "PRON", "NUM", "PUNCT", "SPACE")
        ):
            start_idx += 1
        tokens = tokens[start_idx:]
        if not tokens:
            continue
            
        # 4. Strip trailing stopwords, determiners, prepositions, conjunctions, punctuation
        end_idx = len(tokens)
        while end_idx > 0 and (
            tokens[end_idx - 1].text.lower() in stopwords
            or tokens[end_idx - 1].pos_ in ("DET", "PRON", "ADP", "CCONJ", "PUNCT", "SPACE")
        ):
            end_idx -= 1
        tokens = tokens[:end_idx]
        if not tokens:
            continue
            
        # 5. Check tokens for code / structured syntax
        if any(is_code_token(t.text) for t in tokens):
            continue
            
        # 6. Normalize text (lowercase)
        words = [t.text.lower() for t in tokens if t.text.strip()]
        if not words:
            continue
            
        # Word count constraints: 1 to 6 words
        if len(words) > 6 or len(words) < 1:
            continue
            
        # Drop single-letter words (unless recognized acronym like 'ai')
        if any(len(w) < 2 and w != "ai" for w in words):
            continue
            
        phrase = " ".join(words)
        
        # Must match English alphanumeric and optional internal hyphen
        if not re.match(r"^[a-z][a-z0-9\- ]*[a-z0-9]$", phrase):
            continue
            
        # Must not be a standalone stopword or bare programming keyword
        if phrase in stopwords or phrase in CODE_KEYWORDS:
            continue
            
        phrases.append(phrase)
        
    return phrases

def extract_from_text(text: str, nlp=None) -> list[str]:
    """Extract clean noun phrases from a raw text string."""
    if nlp is None:
        nlp = get_nlp()
    doc = nlp(text)
    return extract_noun_phrases(doc)
