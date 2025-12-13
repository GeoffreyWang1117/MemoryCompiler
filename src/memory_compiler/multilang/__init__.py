"""Multi-language support for memory extraction.

Provides language detection, multi-lingual NER, and
cross-lingual entity resolution capabilities.
"""

from memory_compiler.multilang.detector import (
    LanguageDetector,
    DetectionResult,
    detect_language,
)
from memory_compiler.multilang.extractor import (
    MultilingualExtractor,
    LanguageConfig,
)
from memory_compiler.multilang.tokenizer import (
    MultilingualTokenizer,
    get_tokenizer,
)
from memory_compiler.multilang.transliteration import (
    Transliterator,
    transliterate,
)

__all__ = [
    "LanguageDetector",
    "DetectionResult",
    "detect_language",
    "MultilingualExtractor",
    "LanguageConfig",
    "MultilingualTokenizer",
    "get_tokenizer",
    "Transliterator",
    "transliterate",
]
