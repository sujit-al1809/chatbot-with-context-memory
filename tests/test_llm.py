"""
Tests for the LLM Module.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.llm import get_llm


class TestLLM:
    """Tests for LLM initialization."""

    def test_fake_llm(self):
        """Test fake LLM initialization."""
        llm = get_llm(provider="fake")
        assert llm is not None

    def test_fake_llm_generates_response(self):
        """Test that fake LLM generates a response."""
        llm = get_llm(provider="fake")
        response = llm.invoke("Hello")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_unknown_provider_fallback(self):
        """Test unknown provider falls back gracefully."""
        llm = get_llm(provider="unknown_provider")
        assert llm is not None




class TestUtilities:
    """Tests for utility functions."""

    def test_generate_session_id(self):
        from chatbot.utils import generate_session_id
        sid = generate_session_id()
        assert isinstance(sid, str)
        assert len(sid) == 8

    def test_extract_entities_name(self):
        from chatbot.utils import extract_entities
        entities = extract_entities("My name is Alice")
        assert "name" in entities
        assert entities["name"] == "Alice"

    def test_extract_entities_location(self):
        from chatbot.utils import extract_entities
        entities = extract_entities("I live in New York")
        assert "location" in entities

    def test_extract_entities_preference(self):
        from chatbot.utils import extract_entities
        entities = extract_entities("I really love playing guitar")
        assert "preference" in entities

    def test_extract_entities_empty(self):
        from chatbot.utils import extract_entities
        entities = extract_entities("Hello world")
        assert isinstance(entities, dict)

    def test_count_tokens_approx(self):
        from chatbot.utils import count_tokens_approx
        count = count_tokens_approx("Hello world, this is a test")
        assert isinstance(count, int)
        assert count > 0
