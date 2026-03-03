"""Tests unitarios para retriever y format_context."""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.retriever import format_context


def test_format_context_single_chunk():
    chunks = [{
        'text': 'Some relevant text.',
        'metadata': {
            'title': 'Test Paper',
            'authors': 'Author A',
            'year': 2024
        },
        'distance': 0.1
    }]
    result = format_context(chunks)
    assert 'Test Paper' in result
    assert 'Author A' in result
    assert '2024' in result
    assert 'Some relevant text.' in result


def test_format_context_multiple_chunks():
    chunks = [
        {
            'text': 'Text one.',
            'metadata': {'title': 'Paper 1', 'authors': 'Auth 1', 'year': 2023},
            'distance': 0.1
        },
        {
            'text': 'Text two.',
            'metadata': {'title': 'Paper 2', 'authors': 'Auth 2', 'year': 2024},
            'distance': 0.2
        }
    ]
    result = format_context(chunks)
    assert '[1]' in result
    assert '[2]' in result
    assert '---' in result


def test_format_context_empty():
    result = format_context([])
    assert result == ''
