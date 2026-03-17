"""
Tests for the Chains Module — LangChain chains, sentiment, and intent.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.chains import (
    create_conversation_chain,
    get_response,
    analyze_sentiment,
    detect_intent,
    _clean_response,
)
from chatbot.llm import get_llm
from chatbot.memory import MemoryManager


class TestSentimentAnalysis:
    """Tests for sentiment analysis."""

    def test_positive_sentiment(self):
        result = analyze_sentiment("I love this! It's amazing and wonderful!")
        assert "Positive" in result["label"]
        assert result["score"] > 0
        assert "emoji" in result

    def test_negative_sentiment(self):
        result = analyze_sentiment("This is terrible and horrible. I hate it.")
        assert "Negative" in result["label"]
        assert result["score"] < 0

    def test_neutral_sentiment(self):
        result = analyze_sentiment("The meeting is at 3 PM tomorrow.")
        assert result["label"] == "Neutral"

    def test_sentiment_has_all_fields(self):
        result = analyze_sentiment("Hello")
        assert "score" in result
        assert "label" in result
        assert "emoji" in result
        assert "subjectivity" in result


class TestIntentDetection:
    """Tests for intent detection."""

    def test_greeting(self):
        result = detect_intent("Hello! How are you?")
        assert result["intent"] == "greeting"

    def test_farewell(self):
        result = detect_intent("Goodbye, see you later!")
        assert result["intent"] == "farewell"

    def test_question(self):
        result = detect_intent("What is machine learning?")
        assert result["intent"] == "question"

    def test_help(self):
        result = detect_intent("I need help with this problem")
        assert result["intent"] == "help"

    def test_gratitude(self):
        result = detect_intent("Thank you so much!")
        assert result["intent"] == "gratitude"

    def test_general(self):
        result = detect_intent("xyz abc 123")
        assert result["intent"] == "general"
        assert result["confidence"] == 0.3

    def test_returns_confidence(self):
        result = detect_intent("Hello")
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0


class TestConversationChain:
    """Tests for the conversation chain."""

    def setup_method(self):
        self.llm = get_llm(provider="fake")
        self.mm = MemoryManager(use_persistence=False)
        self.session_id = "test-chain-001"

    def test_create_chain(self):
        memory = self.mm.get_memory(self.session_id)
        chain = create_conversation_chain(self.llm, memory)
        assert chain is not None

    def test_get_response(self):
        memory = self.mm.get_memory(self.session_id)
        chain = create_conversation_chain(self.llm, memory)
        response = get_response(chain, "Hello, how are you?")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_memory_persistence_in_chain(self):
        """Test that memory retains messages across calls."""
        memory = self.mm.get_memory(self.session_id)
        chain = create_conversation_chain(self.llm, memory)

        get_response(chain, "My name is Alice")
        get_response(chain, "What did I just tell you?")

        msg_count = self.mm.get_message_count(self.session_id)
        assert msg_count >= 4  # 2 user + 2 AI messages


class TestResponseCleaning:
    """Tests for response cleaning."""

    def test_strips_whitespace(self):
        assert _clean_response("  hello  ") == "hello"

    def test_removes_excess_newlines(self):
        result = _clean_response("line1\n\n\n\n\nline2")
        assert "\n\n\n" not in result

    def test_removes_system_prompt_leak(self):
        result = _clean_response("You are ContextBot AI blah blah. Actual response.")
        assert "You are ContextBot AI" not in result
