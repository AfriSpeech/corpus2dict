"""corpus2nouns: High-throughput engine for extracting and reducing noun phrases from massive corpora."""

__version__ = "0.1.0"

from corpus2nouns.cleaner import strip_structured_text, is_code_token, chunk_long_text
from corpus2nouns.extractor import extract_noun_phrases, extract_from_text
from corpus2nouns.reducer import reduce_to_pure_nouns
from corpus2nouns.pipeline import CorpusNounPipeline

__all__ = [
    "strip_structured_text",
    "is_code_token",
    "chunk_long_text",
    "extract_noun_phrases",
    "extract_from_text",
    "reduce_to_pure_nouns",
    "CorpusNounPipeline",
    "__version__",
]
