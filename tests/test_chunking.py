"""Tests unitarios para el TokenChunker."""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chunking.chunker import TokenChunker


def test_chunker_returns_list():
    chunker = TokenChunker(chunk_size=50, chunk_overlap=5)
    chunks = chunker.chunk_text("This is a simple test text for chunking.")
    assert isinstance(chunks, list)
    assert len(chunks) >= 1


def test_chunker_chunk_has_required_keys():
    chunker = TokenChunker(chunk_size=50, chunk_overlap=5)
    chunks = chunker.chunk_text("Hello world.")
    assert 'chunk_id' in chunks[0]
    assert 'text' in chunks[0]
    assert 'token_count' in chunks[0]
    assert 'metadata' in chunks[0]


def test_chunker_respects_size():
    chunker = TokenChunker(chunk_size=10, chunk_overlap=2)
    long_text = " ".join(["word"] * 100)
    chunks = chunker.chunk_text(long_text)
    for c in chunks:
        assert c['token_count'] <= 10


def test_chunker_passes_metadata():
    chunker = TokenChunker(chunk_size=50, chunk_overlap=5)
    meta = {'paper_id': 'paper_001', 'title': 'Test'}
    chunks = chunker.chunk_text("Some text here.", metadata=meta)
    assert chunks[0]['metadata']['paper_id'] == 'paper_001'


def test_chunker_count_tokens():
    chunker = TokenChunker()
    count = chunker.count_tokens("hello world")
    assert isinstance(count, int)
    assert count > 0


def test_three_configs_exist():
    from src.chunking.chunker import CHUNKER_SMALL, CHUNKER_DEFAULT, CHUNKER_LARGE
    assert CHUNKER_SMALL.chunk_size == 256
    assert CHUNKER_DEFAULT.chunk_size == 512
    assert CHUNKER_LARGE.chunk_size == 1024
