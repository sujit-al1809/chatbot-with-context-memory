"""
Tests for the Memory Module — LangChain ChatMessageHistory integration.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot.memory import MemoryManager


class TestMemoryManager:
    """Tests for the MemoryManager class."""

    def setup_method(self):
        """Create a non-persistent memory manager for testing."""
        self.mm = MemoryManager(use_persistence=False)
        self.session_id = "test-session-001"

    def test_get_chat_history(self):
        """Test creating/getting chat history."""
        history = self.mm.get_chat_history(self.session_id)
        assert history is not None
        assert len(history.messages) == 0

    def test_add_messages(self):
        """Test adding messages to history."""
        history = self.mm.get_chat_history(self.session_id)
        history.add_user_message("Hello, my name is Alice!")
        history.add_ai_message("Hi Alice! Nice to meet you!")

        assert len(history.messages) == 2
        assert history.messages[0].content == "Hello, my name is Alice!"
        assert history.messages[1].content == "Hi Alice! Nice to meet you!"

    def test_get_message_count(self):
        """Test message counting."""
        history = self.mm.get_chat_history(self.session_id)
        history.add_user_message("Test message 1")
        history.add_ai_message("Response 1")
        history.add_user_message("Test message 2")

        count = self.mm.get_message_count(self.session_id)
        assert count == 3

    def test_get_all_messages(self):
        """Test getting all messages as dicts."""
        history = self.mm.get_chat_history(self.session_id)
        history.add_user_message("What is AI?")
        history.add_ai_message("AI stands for Artificial Intelligence.")

        messages = self.mm.get_all_messages(self.session_id)
        assert len(messages) == 2
        assert messages[0]["role"] == "human"
        assert messages[1]["role"] == "ai"
        assert "AI" in messages[0]["content"]

    def test_get_memory(self):
        """Test creating a ConversationBufferWindowMemory."""
        memory = self.mm.get_memory(self.session_id)
        assert memory is not None
        assert memory.k > 0  # Should have a window size

    def test_clear_session(self):
        """Test clearing session."""
        history = self.mm.get_chat_history(self.session_id)
        history.add_user_message("Data to be cleared")

        self.mm.clear_session(self.session_id)

        # Session should be removed
        assert self.session_id not in self.mm._sessions

    def test_context_summary(self):
        """Test context summary generation."""
        history = self.mm.get_chat_history(self.session_id)
        history.add_user_message("I love machine learning")
        history.add_ai_message("That's great!")
        history.add_user_message("Tell me about neural networks")

        summary = self.mm.get_context_summary(self.session_id)
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "Messages" in summary

    def test_keyword_extraction(self):
        """Test keyword extraction."""
        keywords = self.mm._extract_keywords(
            "machine learning and artificial intelligence are advancing rapidly"
        )
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert "machine" in keywords or "learning" in keywords or "artificial" in keywords

    def test_multiple_sessions(self):
        """Test multiple independent sessions."""
        h1 = self.mm.get_chat_history("session-1")
        h2 = self.mm.get_chat_history("session-2")

        h1.add_user_message("Session 1 message")
        h2.add_user_message("Session 2 message")

        assert self.mm.get_message_count("session-1") == 1
        assert self.mm.get_message_count("session-2") == 1
        assert h1.messages[0].content != h2.messages[0].content

    def test_empty_context_summary(self):
        """Test context summary with no messages."""
        summary = self.mm.get_context_summary("nonexistent-session")
        assert "No conversation history" in summary
