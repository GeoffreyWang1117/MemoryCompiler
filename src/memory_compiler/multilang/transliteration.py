"""Transliteration utilities for cross-script text conversion."""

from __future__ import annotations

from typing import Dict, Optional

from loguru import logger


class Transliterator:
    """Convert text between different scripts.

    Supports romanization of CJK characters and other conversions.
    """

    def __init__(self, use_external: bool = True) -> None:
        """Initialize transliterator.

        Args:
            use_external: Try to use external libraries.
        """
        self._pinyin = None
        self._romaji = None

        if use_external:
            self._init_converters()

    def _init_converters(self) -> None:
        """Initialize conversion libraries."""
        try:
            import pypinyin
            self._pinyin = pypinyin
            logger.debug("Initialized pypinyin for Chinese romanization")
        except ImportError:
            pass

    def to_pinyin(self, text: str, with_tone: bool = True) -> str:
        """Convert Chinese to Pinyin.

        Args:
            text: Chinese text.
            with_tone: Include tone marks.

        Returns:
            Pinyin romanization.
        """
        if self._pinyin:
            from pypinyin import Style, pinyin
            style = Style.TONE if with_tone else Style.NORMAL
            result = pinyin(text, style=style)
            return " ".join([p[0] for p in result])

        # Basic fallback - return original
        return text

    def to_romaji(self, text: str) -> str:
        """Convert Japanese to Romaji.

        Args:
            text: Japanese text.

        Returns:
            Romaji romanization.
        """
        # Basic hiragana/katakana to romaji mapping
        HIRAGANA_ROMAJI = {
            'あ': 'a', 'い': 'i', 'う': 'u', 'え': 'e', 'お': 'o',
            'か': 'ka', 'き': 'ki', 'く': 'ku', 'け': 'ke', 'こ': 'ko',
            'さ': 'sa', 'し': 'shi', 'す': 'su', 'せ': 'se', 'そ': 'so',
            'た': 'ta', 'ち': 'chi', 'つ': 'tsu', 'て': 'te', 'と': 'to',
            'な': 'na', 'に': 'ni', 'ぬ': 'nu', 'ね': 'ne', 'の': 'no',
            'は': 'ha', 'ひ': 'hi', 'ふ': 'fu', 'へ': 'he', 'ほ': 'ho',
            'ま': 'ma', 'み': 'mi', 'む': 'mu', 'め': 'me', 'も': 'mo',
            'や': 'ya', 'ゆ': 'yu', 'よ': 'yo',
            'ら': 'ra', 'り': 'ri', 'る': 'ru', 'れ': 're', 'ろ': 'ro',
            'わ': 'wa', 'を': 'wo', 'ん': 'n',
        }

        result = []
        for char in text:
            if char in HIRAGANA_ROMAJI:
                result.append(HIRAGANA_ROMAJI[char])
            else:
                result.append(char)

        return "".join(result)

    def transliterate(
        self,
        text: str,
        source_lang: str,
        target_script: str = "latin",
    ) -> str:
        """Transliterate text to target script.

        Args:
            text: Source text.
            source_lang: Source language code.
            target_script: Target script (latin, etc.).

        Returns:
            Transliterated text.
        """
        if target_script == "latin":
            if source_lang == "zh":
                return self.to_pinyin(text)
            elif source_lang == "ja":
                return self.to_romaji(text)

        return text


# Global instance
_transliterator: Optional[Transliterator] = None


def transliterate(text: str, source_lang: str, target_script: str = "latin") -> str:
    """Transliterate text using global transliterator."""
    global _transliterator
    if _transliterator is None:
        _transliterator = Transliterator()
    return _transliterator.transliterate(text, source_lang, target_script)
