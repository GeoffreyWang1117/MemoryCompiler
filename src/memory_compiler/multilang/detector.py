"""Language detection for multi-lingual support."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


class Language(Enum):
    """Supported languages."""

    ENGLISH = "en"
    CHINESE = "zh"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    KOREAN = "ko"
    RUSSIAN = "ru"
    ARABIC = "ar"
    PORTUGUESE = "pt"
    ITALIAN = "it"
    DUTCH = "nl"
    HINDI = "hi"
    THAI = "th"
    VIETNAMESE = "vi"
    UNKNOWN = "unknown"

    @classmethod
    def from_code(cls, code: str) -> "Language":
        """Get language from ISO code."""
        code = code.lower()[:2]
        for lang in cls:
            if lang.value == code:
                return lang
        return cls.UNKNOWN


@dataclass
class DetectionResult:
    """Language detection result.

    Attributes:
        language: Detected primary language.
        confidence: Detection confidence (0-1).
        alternatives: Other possible languages.
        is_multilingual: Whether text contains multiple languages.
        detected_scripts: Scripts found in text.
    """

    language: Language
    confidence: float
    alternatives: List[Tuple[Language, float]] = field(default_factory=list)
    is_multilingual: bool = False
    detected_scripts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "language": self.language.value,
            "confidence": self.confidence,
            "alternatives": [(l.value, c) for l, c in self.alternatives],
            "is_multilingual": self.is_multilingual,
            "detected_scripts": self.detected_scripts,
        }


class LanguageDetector:
    """Detect language of text.

    Uses multiple strategies for robust language detection:
    1. Script/character-based detection
    2. N-gram analysis
    3. Optional external libraries (langdetect, fasttext)

    Example:
        >>> detector = LanguageDetector()
        >>> result = detector.detect("Hello, how are you?")
        >>> print(result.language)  # Language.ENGLISH
    """

    # Unicode script ranges
    SCRIPT_RANGES = {
        "latin": (0x0000, 0x024F),
        "cyrillic": (0x0400, 0x04FF),
        "arabic": (0x0600, 0x06FF),
        "devanagari": (0x0900, 0x097F),
        "thai": (0x0E00, 0x0E7F),
        "cjk": (0x4E00, 0x9FFF),
        "hangul": (0xAC00, 0xD7AF),
        "hiragana": (0x3040, 0x309F),
        "katakana": (0x30A0, 0x30FF),
    }

    # Script to primary language mapping
    SCRIPT_LANGUAGES = {
        "cyrillic": Language.RUSSIAN,
        "arabic": Language.ARABIC,
        "devanagari": Language.HINDI,
        "thai": Language.THAI,
        "hangul": Language.KOREAN,
        "hiragana": Language.JAPANESE,
        "katakana": Language.JAPANESE,
    }

    # Common words for language identification
    LANGUAGE_MARKERS = {
        Language.ENGLISH: ["the", "is", "are", "and", "to", "of", "in", "that", "it", "for"],
        Language.SPANISH: ["el", "la", "de", "que", "en", "es", "los", "las", "del", "un"],
        Language.FRENCH: ["le", "la", "de", "et", "les", "des", "en", "un", "une", "que"],
        Language.GERMAN: ["der", "die", "und", "in", "den", "von", "das", "ist", "zu", "mit"],
        Language.PORTUGUESE: ["de", "que", "em", "para", "com", "uma", "os", "da", "do", "ao"],
        Language.ITALIAN: ["di", "che", "il", "la", "per", "un", "una", "con", "del", "della"],
        Language.DUTCH: ["de", "het", "een", "en", "van", "in", "is", "dat", "op", "te"],
        Language.CHINESE: ["的", "是", "在", "了", "和", "我", "有", "他", "这", "为"],
        Language.JAPANESE: ["の", "は", "が", "を", "に", "で", "と", "です", "ます", "た"],
        Language.KOREAN: ["을", "를", "이", "가", "의", "에", "는", "은", "다", "고"],
    }

    def __init__(
        self,
        use_external: bool = True,
        default_language: Language = Language.ENGLISH,
        min_confidence: float = 0.5,
    ) -> None:
        """Initialize detector.

        Args:
            use_external: Try to use external libraries.
            default_language: Default when detection fails.
            min_confidence: Minimum confidence threshold.
        """
        self.default_language = default_language
        self.min_confidence = min_confidence

        self._external_detector = None
        if use_external:
            self._init_external_detector()

    def _init_external_detector(self) -> None:
        """Initialize external detector if available."""
        try:
            import langdetect
            self._external_detector = "langdetect"
            logger.debug("Using langdetect for language detection")
        except ImportError:
            try:
                import fasttext
                self._external_detector = "fasttext"
                logger.debug("Using fasttext for language detection")
            except ImportError:
                logger.debug("No external language detector available")

    def detect(self, text: str) -> DetectionResult:
        """Detect the language of text.

        Args:
            text: Text to analyze.

        Returns:
            Detection result with language and confidence.
        """
        if not text or not text.strip():
            return DetectionResult(
                language=self.default_language,
                confidence=0.0,
            )

        # Detect scripts first
        scripts = self._detect_scripts(text)

        # Check for script-based detection (non-Latin scripts)
        script_result = self._detect_by_script(text, scripts)
        if script_result:
            return script_result

        # Try external detector
        if self._external_detector:
            ext_result = self._detect_external(text)
            if ext_result and ext_result.confidence >= self.min_confidence:
                ext_result.detected_scripts = scripts
                return ext_result

        # Fall back to internal detection
        internal_result = self._detect_internal(text)
        internal_result.detected_scripts = scripts

        return internal_result

    def _detect_scripts(self, text: str) -> List[str]:
        """Detect scripts present in text."""
        scripts = set()

        for char in text:
            code = ord(char)
            for script, (start, end) in self.SCRIPT_RANGES.items():
                if start <= code <= end:
                    scripts.add(script)
                    break

        return list(scripts)

    def _detect_by_script(
        self,
        text: str,
        scripts: List[str],
    ) -> Optional[DetectionResult]:
        """Detect language based on script."""
        # Filter out latin - need further analysis
        non_latin = [s for s in scripts if s != "latin"]

        if not non_latin:
            return None

        # Count characters per script
        script_counts: Dict[str, int] = {s: 0 for s in non_latin}
        total = 0

        for char in text:
            if char.isspace():
                continue
            total += 1
            code = ord(char)
            for script in non_latin:
                start, end = self.SCRIPT_RANGES[script]
                if start <= code <= end:
                    script_counts[script] += 1
                    break

        if total == 0:
            return None

        # Find dominant script
        dominant = max(script_counts.items(), key=lambda x: x[1])
        script_name, count = dominant
        confidence = count / total

        if confidence < 0.3:
            return None

        # CJK needs special handling
        if script_name == "cjk":
            return self._detect_cjk_language(text, confidence)

        # Map script to language
        lang = self.SCRIPT_LANGUAGES.get(script_name)
        if lang:
            return DetectionResult(
                language=lang,
                confidence=confidence,
                detected_scripts=scripts,
            )

        return None

    def _detect_cjk_language(
        self,
        text: str,
        base_confidence: float,
    ) -> DetectionResult:
        """Distinguish between Chinese and Japanese with CJK characters."""
        # Count Japanese-specific characters
        hiragana_count = 0
        katakana_count = 0

        for char in text:
            code = ord(char)
            if 0x3040 <= code <= 0x309F:
                hiragana_count += 1
            elif 0x30A0 <= code <= 0x30FF:
                katakana_count += 1

        total_kana = hiragana_count + katakana_count

        if total_kana > len(text) * 0.1:
            # Significant kana presence -> Japanese
            return DetectionResult(
                language=Language.JAPANESE,
                confidence=base_confidence,
                alternatives=[(Language.CHINESE, base_confidence * 0.5)],
            )

        # Default to Chinese for CJK without kana
        return DetectionResult(
            language=Language.CHINESE,
            confidence=base_confidence,
            alternatives=[(Language.JAPANESE, base_confidence * 0.3)],
        )

    def _detect_external(self, text: str) -> Optional[DetectionResult]:
        """Detect using external library."""
        if self._external_detector == "langdetect":
            return self._detect_langdetect(text)
        elif self._external_detector == "fasttext":
            return self._detect_fasttext(text)
        return None

    def _detect_langdetect(self, text: str) -> Optional[DetectionResult]:
        """Detect using langdetect library."""
        try:
            import langdetect
            from langdetect import detect_langs

            langs = detect_langs(text)
            if not langs:
                return None

            primary = langs[0]
            language = Language.from_code(primary.lang)

            alternatives = [
                (Language.from_code(l.lang), l.prob)
                for l in langs[1:3]
            ]

            return DetectionResult(
                language=language,
                confidence=primary.prob,
                alternatives=alternatives,
                is_multilingual=len(langs) > 1 and langs[1].prob > 0.2,
            )

        except Exception as e:
            logger.debug(f"langdetect error: {e}")
            return None

    def _detect_fasttext(self, text: str) -> Optional[DetectionResult]:
        """Detect using fasttext library."""
        try:
            # fasttext requires model download
            # This is a placeholder for the actual implementation
            return None
        except Exception as e:
            logger.debug(f"fasttext error: {e}")
            return None

    def _detect_internal(self, text: str) -> DetectionResult:
        """Internal detection using word markers."""
        # Normalize text
        words = text.lower().split()
        word_set = set(words)

        # Score each language
        scores: Dict[Language, float] = {}

        for lang, markers in self.LANGUAGE_MARKERS.items():
            overlap = len(word_set & set(markers))
            if overlap > 0:
                scores[lang] = overlap / len(markers)

        if not scores:
            return DetectionResult(
                language=self.default_language,
                confidence=0.3,
            )

        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        primary = sorted_scores[0]
        alternatives = [(l, s) for l, s in sorted_scores[1:3]]

        return DetectionResult(
            language=primary[0],
            confidence=min(primary[1] * 2, 1.0),  # Scale up
            alternatives=alternatives,
            is_multilingual=len(alternatives) > 0 and alternatives[0][1] > 0.3,
        )

    def detect_segments(
        self,
        text: str,
        min_segment_length: int = 20,
    ) -> List[Tuple[str, DetectionResult]]:
        """Detect language of text segments.

        Useful for mixed-language texts.

        Args:
            text: Text to analyze.
            min_segment_length: Minimum segment length.

        Returns:
            List of (segment, detection_result) tuples.
        """
        # Simple sentence-based segmentation
        sentences = re.split(r'[.!?。！？\n]+', text)
        results = []

        current_segment = ""
        current_lang = None

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            detection = self.detect(sentence)

            if current_lang == detection.language:
                current_segment += " " + sentence
            else:
                if current_segment and len(current_segment) >= min_segment_length:
                    results.append((current_segment, self.detect(current_segment)))
                current_segment = sentence
                current_lang = detection.language

        if current_segment and len(current_segment) >= min_segment_length:
            results.append((current_segment, self.detect(current_segment)))

        return results


# Global detector instance
_default_detector: Optional[LanguageDetector] = None


def detect_language(text: str) -> DetectionResult:
    """Detect language of text using global detector.

    Args:
        text: Text to analyze.

    Returns:
        Detection result.
    """
    global _default_detector
    if _default_detector is None:
        _default_detector = LanguageDetector()
    return _default_detector.detect(text)
