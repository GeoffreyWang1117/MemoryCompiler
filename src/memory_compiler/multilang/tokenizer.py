"""Multi-lingual tokenization for memory extraction."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple

from loguru import logger

from memory_compiler.multilang.detector import Language, LanguageDetector


@dataclass
class Token:
    """A text token.

    Attributes:
        text: Token text.
        start: Start character offset.
        end: End character offset.
        pos: Part-of-speech tag.
        is_entity: Whether token is an entity.
        entity_type: Entity type if applicable.
        lemma: Base form of word.
        language: Token language.
    """

    text: str
    start: int
    end: int
    pos: str = ""
    is_entity: bool = False
    entity_type: str = ""
    lemma: str = ""
    language: str = "en"

    @property
    def length(self) -> int:
        """Token length."""
        return self.end - self.start


class TokenizerType(Enum):
    """Types of tokenizers."""

    WHITESPACE = "whitespace"
    REGEX = "regex"
    SPACY = "spacy"
    JIEBA = "jieba"  # Chinese
    MECAB = "mecab"  # Japanese
    KONLPY = "konlpy"  # Korean
    STANZA = "stanza"  # Multi-language


class BaseTokenizer(ABC):
    """Base tokenizer interface."""

    @abstractmethod
    def tokenize(self, text: str) -> List[Token]:
        """Tokenize text into tokens.

        Args:
            text: Text to tokenize.

        Returns:
            List of tokens.
        """
        pass

    def tokenize_sentences(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Text to split.

        Returns:
            List of sentences.
        """
        # Default sentence splitting
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class WhitespaceTokenizer(BaseTokenizer):
    """Simple whitespace-based tokenizer.

    Works for space-separated languages like English, Spanish, etc.
    """

    def __init__(self, lowercase: bool = False) -> None:
        """Initialize tokenizer.

        Args:
            lowercase: Whether to lowercase tokens.
        """
        self.lowercase = lowercase

    def tokenize(self, text: str) -> List[Token]:
        """Tokenize by whitespace and punctuation."""
        tokens = []
        # Pattern to match words and punctuation
        pattern = r'\b\w+\b|[^\s\w]'

        for match in re.finditer(pattern, text):
            token_text = match.group()
            if self.lowercase:
                token_text = token_text.lower()

            tokens.append(Token(
                text=token_text,
                start=match.start(),
                end=match.end(),
            ))

        return tokens


class ChineseTokenizer(BaseTokenizer):
    """Chinese tokenizer using jieba.

    Provides word segmentation for Chinese text.
    """

    def __init__(self, cut_all: bool = False) -> None:
        """Initialize Chinese tokenizer.

        Args:
            cut_all: Full segmentation mode.
        """
        self.cut_all = cut_all
        self._jieba = None
        self._init_jieba()

    def _init_jieba(self) -> None:
        """Initialize jieba."""
        try:
            import jieba
            self._jieba = jieba
            logger.debug("Initialized jieba Chinese tokenizer")
        except ImportError:
            logger.warning("jieba not installed, falling back to character tokenization")

    def tokenize(self, text: str) -> List[Token]:
        """Tokenize Chinese text."""
        tokens = []

        if self._jieba:
            # Use jieba segmentation
            words = list(self._jieba.cut(text, cut_all=self.cut_all))
            offset = 0

            for word in words:
                if word.strip():
                    start = text.find(word, offset)
                    if start == -1:
                        start = offset
                    end = start + len(word)

                    tokens.append(Token(
                        text=word,
                        start=start,
                        end=end,
                        language="zh",
                    ))

                    offset = end
        else:
            # Character-level fallback
            for i, char in enumerate(text):
                if not char.isspace():
                    tokens.append(Token(
                        text=char,
                        start=i,
                        end=i + 1,
                        language="zh",
                    ))

        return tokens

    def tokenize_sentences(self, text: str) -> List[str]:
        """Split Chinese text into sentences."""
        sentences = re.split(r'[。！？\n]+', text)
        return [s.strip() for s in sentences if s.strip()]


class JapaneseTokenizer(BaseTokenizer):
    """Japanese tokenizer.

    Uses MeCab or basic segmentation.
    """

    def __init__(self) -> None:
        """Initialize Japanese tokenizer."""
        self._mecab = None
        self._init_mecab()

    def _init_mecab(self) -> None:
        """Initialize MeCab."""
        try:
            import MeCab
            self._mecab = MeCab.Tagger()
            logger.debug("Initialized MeCab Japanese tokenizer")
        except ImportError:
            logger.warning("MeCab not installed, using basic tokenization")

    def tokenize(self, text: str) -> List[Token]:
        """Tokenize Japanese text."""
        tokens = []

        if self._mecab:
            self._mecab.parse("")  # Initialize
            node = self._mecab.parseToNode(text)
            offset = 0

            while node:
                if node.surface:
                    # Get POS from feature
                    features = node.feature.split(",")
                    pos = features[0] if features else ""

                    start = text.find(node.surface, offset)
                    if start == -1:
                        start = offset
                    end = start + len(node.surface)

                    tokens.append(Token(
                        text=node.surface,
                        start=start,
                        end=end,
                        pos=pos,
                        language="ja",
                    ))

                    offset = end

                node = node.next
        else:
            # Basic fallback - split on common boundaries
            pattern = r'[\u3040-\u309F]+|[\u30A0-\u30FF]+|[\u4E00-\u9FFF]+|\w+'

            for match in re.finditer(pattern, text):
                tokens.append(Token(
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    language="ja",
                ))

        return tokens

    def tokenize_sentences(self, text: str) -> List[str]:
        """Split Japanese text into sentences."""
        sentences = re.split(r'[。！？\n]+', text)
        return [s.strip() for s in sentences if s.strip()]


class KoreanTokenizer(BaseTokenizer):
    """Korean tokenizer.

    Uses KoNLPy or basic segmentation.
    """

    def __init__(self, tagger: str = "okt") -> None:
        """Initialize Korean tokenizer.

        Args:
            tagger: KoNLPy tagger to use (okt, komoran, kkma).
        """
        self._tagger = None
        self._init_tagger(tagger)

    def _init_tagger(self, tagger: str) -> None:
        """Initialize KoNLPy tagger."""
        try:
            if tagger == "okt":
                from konlpy.tag import Okt
                self._tagger = Okt()
            elif tagger == "komoran":
                from konlpy.tag import Komoran
                self._tagger = Komoran()
            elif tagger == "kkma":
                from konlpy.tag import Kkma
                self._tagger = Kkma()
            logger.debug(f"Initialized KoNLPy {tagger} tokenizer")
        except ImportError:
            logger.warning("KoNLPy not installed, using basic tokenization")

    def tokenize(self, text: str) -> List[Token]:
        """Tokenize Korean text."""
        tokens = []

        if self._tagger:
            try:
                morphs = self._tagger.pos(text)
                offset = 0

                for word, pos in morphs:
                    start = text.find(word, offset)
                    if start == -1:
                        start = offset
                    end = start + len(word)

                    tokens.append(Token(
                        text=word,
                        start=start,
                        end=end,
                        pos=pos,
                        language="ko",
                    ))

                    offset = end
            except Exception as e:
                logger.warning(f"KoNLPy error: {e}")
                return self._basic_tokenize(text)
        else:
            return self._basic_tokenize(text)

        return tokens

    def _basic_tokenize(self, text: str) -> List[Token]:
        """Basic Korean tokenization."""
        tokens = []
        pattern = r'[\uAC00-\uD7AF]+|\w+'

        for match in re.finditer(pattern, text):
            tokens.append(Token(
                text=match.group(),
                start=match.start(),
                end=match.end(),
                language="ko",
            ))

        return tokens


class SpacyTokenizer(BaseTokenizer):
    """spaCy-based tokenizer with NER support.

    Provides rich tokenization with POS tagging and entity recognition.
    """

    # Language model mapping
    MODELS = {
        "en": "en_core_web_sm",
        "es": "es_core_news_sm",
        "fr": "fr_core_news_sm",
        "de": "de_core_news_sm",
        "it": "it_core_news_sm",
        "pt": "pt_core_news_sm",
        "nl": "nl_core_news_sm",
        "ru": "ru_core_news_sm",
        "zh": "zh_core_web_sm",
        "ja": "ja_core_news_sm",
    }

    def __init__(
        self,
        language: str = "en",
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize spaCy tokenizer.

        Args:
            language: Language code.
            model_name: Specific model name to use.
        """
        self.language = language
        self._nlp = None
        self._init_spacy(language, model_name)

    def _init_spacy(self, language: str, model_name: Optional[str]) -> None:
        """Initialize spaCy."""
        try:
            import spacy

            model = model_name or self.MODELS.get(language, "en_core_web_sm")

            try:
                self._nlp = spacy.load(model)
                logger.debug(f"Loaded spaCy model: {model}")
            except OSError:
                logger.warning(f"Model {model} not found, using blank model")
                self._nlp = spacy.blank(language)

        except ImportError:
            logger.warning("spaCy not installed")

    def tokenize(self, text: str) -> List[Token]:
        """Tokenize with spaCy."""
        if not self._nlp:
            return WhitespaceTokenizer().tokenize(text)

        doc = self._nlp(text)
        tokens = []

        for token in doc:
            tok = Token(
                text=token.text,
                start=token.idx,
                end=token.idx + len(token.text),
                pos=token.pos_,
                lemma=token.lemma_,
                language=self.language,
            )

            # Check if part of entity
            if token.ent_type_:
                tok.is_entity = True
                tok.entity_type = token.ent_type_

            tokens.append(tok)

        return tokens

    def tokenize_sentences(self, text: str) -> List[str]:
        """Split with spaCy sentence detection."""
        if not self._nlp:
            return super().tokenize_sentences(text)

        doc = self._nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def extract_entities(self, text: str) -> List[Tuple[str, str, int, int]]:
        """Extract named entities.

        Returns:
            List of (text, entity_type, start, end) tuples.
        """
        if not self._nlp:
            return []

        doc = self._nlp(text)
        return [
            (ent.text, ent.label_, ent.start_char, ent.end_char)
            for ent in doc.ents
        ]


class MultilingualTokenizer:
    """Tokenizer that automatically handles multiple languages.

    Detects language and uses appropriate tokenizer.

    Example:
        >>> tokenizer = MultilingualTokenizer()
        >>> tokens = tokenizer.tokenize("Hello world! 你好世界！")
    """

    def __init__(
        self,
        default_language: str = "en",
        use_spacy: bool = True,
    ) -> None:
        """Initialize multi-lingual tokenizer.

        Args:
            default_language: Default language.
            use_spacy: Whether to use spaCy when available.
        """
        self.default_language = default_language
        self.use_spacy = use_spacy

        self._detector = LanguageDetector()
        self._tokenizers: Dict[str, BaseTokenizer] = {}

    def _get_tokenizer(self, language: str) -> BaseTokenizer:
        """Get tokenizer for language."""
        if language not in self._tokenizers:
            self._tokenizers[language] = self._create_tokenizer(language)
        return self._tokenizers[language]

    def _create_tokenizer(self, language: str) -> BaseTokenizer:
        """Create tokenizer for language."""
        if language == "zh":
            return ChineseTokenizer()
        elif language == "ja":
            return JapaneseTokenizer()
        elif language == "ko":
            return KoreanTokenizer()
        elif self.use_spacy:
            return SpacyTokenizer(language)
        else:
            return WhitespaceTokenizer()

    def tokenize(self, text: str, language: Optional[str] = None) -> List[Token]:
        """Tokenize text with automatic language detection.

        Args:
            text: Text to tokenize.
            language: Override language detection.

        Returns:
            List of tokens.
        """
        if language is None:
            result = self._detector.detect(text)
            language = result.language.value

        tokenizer = self._get_tokenizer(language)
        return tokenizer.tokenize(text)

    def tokenize_sentences(
        self,
        text: str,
        language: Optional[str] = None,
    ) -> List[str]:
        """Split text into sentences.

        Args:
            text: Text to split.
            language: Override language.

        Returns:
            List of sentences.
        """
        if language is None:
            result = self._detector.detect(text)
            language = result.language.value

        tokenizer = self._get_tokenizer(language)
        return tokenizer.tokenize_sentences(text)

    def tokenize_mixed(self, text: str) -> List[Tuple[str, List[Token]]]:
        """Tokenize mixed-language text.

        Returns segments with their tokens.

        Args:
            text: Mixed-language text.

        Returns:
            List of (language_code, tokens) tuples.
        """
        segments = self._detector.detect_segments(text)
        results = []

        for segment, detection in segments:
            language = detection.language.value
            tokenizer = self._get_tokenizer(language)
            tokens = tokenizer.tokenize(segment)
            results.append((language, tokens))

        return results


# Global tokenizer
_global_tokenizer: Optional[MultilingualTokenizer] = None


def get_tokenizer() -> MultilingualTokenizer:
    """Get global tokenizer instance."""
    global _global_tokenizer
    if _global_tokenizer is None:
        _global_tokenizer = MultilingualTokenizer()
    return _global_tokenizer
