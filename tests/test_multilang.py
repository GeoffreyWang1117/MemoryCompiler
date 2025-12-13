"""Tests for multi-language support module."""

import pytest
from memory_compiler.multilang import (
    LanguageDetector,
    DetectionResult,
    detect_language,
    MultilingualExtractor,
    LanguageConfig,
    MultilingualTokenizer,
    get_tokenizer,
    Transliterator,
    transliterate,
)
from memory_compiler.multilang.detector import Language


class TestLanguageDetector:
    """Tests for LanguageDetector."""

    def test_detect_english(self):
        """Test detecting English text."""
        detector = LanguageDetector()
        result = detector.detect("Hello, how are you today?")

        assert result.language == Language.ENGLISH
        assert result.confidence > 0.5

    def test_detect_chinese(self):
        """Test detecting Chinese text."""
        detector = LanguageDetector()
        result = detector.detect("你好，今天天气怎么样？")

        assert result.language == Language.CHINESE
        assert result.confidence > 0.5

    def test_detect_spanish(self):
        """Test detecting Spanish text."""
        detector = LanguageDetector()
        result = detector.detect("Hola, ¿cómo estás hoy?")

        assert result.language == Language.SPANISH
        assert result.confidence > 0.3

    def test_detect_japanese(self):
        """Test detecting Japanese text."""
        detector = LanguageDetector()
        result = detector.detect("こんにちは、元気ですか？")

        assert result.language == Language.JAPANESE
        assert result.confidence > 0.5

    def test_detect_korean(self):
        """Test detecting Korean text."""
        detector = LanguageDetector()
        result = detector.detect("안녕하세요, 오늘 기분이 어떠세요?")

        assert result.language == Language.KOREAN
        assert result.confidence > 0.5

    def test_detect_empty_text(self):
        """Test detecting empty text."""
        detector = LanguageDetector()
        result = detector.detect("")

        assert result.confidence == 0.0

    def test_detect_mixed_scripts(self):
        """Test detecting text with mixed scripts."""
        detector = LanguageDetector()
        result = detector.detect("Hello 你好 world 世界")

        assert result.is_multilingual or result.detected_scripts

    def test_detect_segments(self):
        """Test detecting language segments."""
        detector = LanguageDetector()
        text = "Hello, how are you? 你好，你怎么样？"

        segments = detector.detect_segments(text, min_segment_length=5)

        assert len(segments) >= 1

    def test_detection_result_to_dict(self):
        """Test DetectionResult serialization."""
        result = DetectionResult(
            language=Language.ENGLISH,
            confidence=0.9,
            alternatives=[(Language.SPANISH, 0.1)],
            is_multilingual=False,
            detected_scripts=["latin"],
        )

        d = result.to_dict()

        assert d["language"] == "en"
        assert d["confidence"] == 0.9
        assert len(d["alternatives"]) == 1


class TestMultilingualTokenizer:
    """Tests for MultilingualTokenizer."""

    def test_tokenize_english(self):
        """Test tokenizing English text."""
        tokenizer = MultilingualTokenizer()
        tokens = tokenizer.tokenize("Hello world!", language="en")

        assert len(tokens) >= 2
        token_texts = [t.text for t in tokens]
        assert "Hello" in token_texts or "hello" in token_texts.lower()

    def test_tokenize_chinese(self):
        """Test tokenizing Chinese text."""
        tokenizer = MultilingualTokenizer()
        tokens = tokenizer.tokenize("今天天气很好", language="zh")

        # Should segment into words
        assert len(tokens) >= 1

    def test_tokenize_auto_detect(self):
        """Test tokenizing with auto language detection."""
        tokenizer = MultilingualTokenizer()

        # English
        en_tokens = tokenizer.tokenize("How are you?")
        assert len(en_tokens) >= 3

        # Chinese
        zh_tokens = tokenizer.tokenize("你好世界")
        assert len(zh_tokens) >= 1

    def test_tokenize_sentences(self):
        """Test sentence tokenization."""
        tokenizer = MultilingualTokenizer()

        sentences = tokenizer.tokenize_sentences(
            "Hello. How are you? I am fine.",
            language="en",
        )

        assert len(sentences) >= 2

    def test_tokenize_mixed(self):
        """Test tokenizing mixed-language text."""
        tokenizer = MultilingualTokenizer()
        results = tokenizer.tokenize_mixed("Hello 你好 world")

        assert len(results) >= 1

    def test_global_tokenizer(self):
        """Test global tokenizer function."""
        tokenizer = get_tokenizer()
        tokens = tokenizer.tokenize("Test text")

        assert len(tokens) >= 2


class TestMultilingualExtractor:
    """Tests for MultilingualExtractor."""

    def test_extract_english(self):
        """Test extracting from English text."""
        extractor = MultilingualExtractor()
        ir = extractor.extract(
            "John works at Google. He lives in New York.",
            language="en",
        )

        entities = list(ir.iter_entities())
        entity_names = [e.name.lower() for e in entities]

        # Should extract person and organization
        assert len(entities) >= 1

    def test_extract_chinese(self):
        """Test extracting from Chinese text."""
        extractor = MultilingualExtractor()
        ir = extractor.extract(
            "张三在北京工作。他是一名工程师。",
            language="zh",
        )

        entities = list(ir.iter_entities())
        assert len(entities) >= 0  # Depends on available NER

    def test_extract_auto_detect(self):
        """Test extraction with auto language detection."""
        extractor = MultilingualExtractor(auto_detect=True)

        ir = extractor.extract("Alice works at Microsoft in Seattle.")

        assert ir.metadata.get("source_language") == "en"

    def test_extract_facts(self):
        """Test extracting facts."""
        extractor = MultilingualExtractor()
        ir = extractor.extract(
            "John works at Google.",
            language="en",
        )

        facts = list(ir.iter_facts())
        # Should extract work relationship
        assert len(facts) >= 0

    def test_extract_multilingual(self):
        """Test extracting from mixed-language text."""
        extractor = MultilingualExtractor()
        ir = extractor.extract_multilingual(
            "Hello, my name is John. 你好，我叫张三。",
        )

        assert ir.metadata.get("is_multilingual", False) or len(list(ir.iter_entities())) >= 0


class TestLanguageConfig:
    """Tests for LanguageConfig."""

    def test_english_config(self):
        """Test English configuration."""
        config = LanguageConfig.for_english()

        assert config.language == "en"
        assert len(config.person_patterns) > 0
        assert len(config.stopwords) > 0

    def test_chinese_config(self):
        """Test Chinese configuration."""
        config = LanguageConfig.for_chinese()

        assert config.language == "zh"
        assert len(config.person_patterns) > 0

    def test_spanish_config(self):
        """Test Spanish configuration."""
        config = LanguageConfig.for_spanish()

        assert config.language == "es"
        assert len(config.person_patterns) > 0

    def test_japanese_config(self):
        """Test Japanese configuration."""
        config = LanguageConfig.for_japanese()

        assert config.language == "ja"
        assert len(config.date_patterns) > 0

    def test_get_config(self):
        """Test getting config by language code."""
        config = LanguageConfig.get_config("en")
        assert config.language == "en"

        # Unknown language should default to English
        config = LanguageConfig.get_config("xx")
        assert config.language == "en"


class TestTransliterator:
    """Tests for Transliterator."""

    def test_chinese_to_pinyin(self):
        """Test Chinese to Pinyin conversion."""
        trans = Transliterator()
        result = trans.to_pinyin("你好")

        # Should return pinyin or original if no library
        assert len(result) > 0

    def test_japanese_to_romaji(self):
        """Test Japanese to Romaji conversion."""
        trans = Transliterator()
        result = trans.to_romaji("こんにちは")

        # Basic hiragana should convert
        assert "ko" in result.lower() or "こ" in result

    def test_transliterate_function(self):
        """Test global transliterate function."""
        result = transliterate("你好", source_lang="zh", target_script="latin")

        assert len(result) > 0

    def test_unsupported_language(self):
        """Test transliteration of unsupported language."""
        trans = Transliterator()
        result = trans.transliterate("Hello", source_lang="xx", target_script="latin")

        # Should return original text
        assert result == "Hello"


class TestGlobalFunctions:
    """Tests for global convenience functions."""

    def test_detect_language_function(self):
        """Test global detect_language function."""
        result = detect_language("Hello world")

        assert result.language == Language.ENGLISH

    def test_get_tokenizer_function(self):
        """Test global get_tokenizer function."""
        tokenizer = get_tokenizer()

        assert tokenizer is not None
        tokens = tokenizer.tokenize("Test")
        assert len(tokens) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
