"""Tests unitarios para los módulos de ingestión."""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.text_cleaner import clean_extracted_text


def test_clean_removes_extra_whitespace():
    raw = "hello   world\n\nfoo   bar"
    result = clean_extracted_text(raw)
    assert "  " not in result


def test_clean_fixes_hyphenated_words():
    raw = "algo- rithm"
    result = clean_extracted_text(raw)
    assert "algorithm" in result


def test_clean_replaces_curly_quotes():
    raw = "\u201chello\u201d"
    result = clean_extracted_text(raw)
    assert '"hello"' in result


def test_clean_strips_whitespace():
    raw = "  hello world  "
    result = clean_extracted_text(raw)
    assert result == result.strip()


def test_clean_returns_string():
    result = clean_extracted_text("some text")
    assert isinstance(result, str)
