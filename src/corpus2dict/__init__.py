"""corpus2dict: High-throughput engine for extracting, reducing, and translating noun lexicons and gloss-mt dictionaries from massive corpora."""

__version__ = "0.1.0"

from corpus2dict.cleaner import strip_structured_text, is_code_token, chunk_long_text
from corpus2dict.extractor import extract_noun_phrases, extract_from_text
from corpus2dict.reducer import reduce_to_pure_nouns, classify_noun_type
from corpus2dict.pipeline import CorpusNounPipeline

__all__ = [
    "strip_structured_text",
    "is_code_token",
    "chunk_long_text",
    "extract_noun_phrases",
    "extract_from_text",
    "reduce_to_pure_nouns",
    "classify_noun_type",
    "CorpusNounPipeline",
    "__version__",
]
