"""
Memory Module — Conversational memory using LangChain ChatMessageHistory.

Implements:
1. ChatMessageHistory — In-memory conversation history per session
2. ConversationBufferWindowMemory — Sliding window of K recent exchanges
3. ConversationSummaryMemory — Auto-summarized memory for long conversations
4. SQLite persistence — Saves chat history across app restarts
"""
import os
import logging
from typing import Optional
from pathlib import Path

from langchain.memory import (
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory,
    CombinedMemory,
)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import (
    ChatMessageHistory,
    SQLChatMessageHistory,
)

from chatbot.config import MEMORY_WINDOW_SIZE, MAX_TOKEN_LIMIT, CHAT_DB_PATH

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages conversational memory for the chatbot using LangChain.

    Features:
    - Per-session chat history with ChatMessageHistory
    - Sliding window memory (last K messages) for context
    - Optional summary memory for compressing long conversations
    - SQLite persistence for chat history across sessions
    """

    def __init__(self, use_persistence: bool = True):
        """
        Initialize the memory manager.

        Args:
            use_persistence: If True, persists chat history to SQLite
        """
        self.use_persistence = use_persistence
        self._sessions: dict[str, dict] = {}

        # Ensure data directory exists
        if use_persistence:
            Path(CHAT_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
            self.connection_string = f"sqlite:///{CHAT_DB_PATH}"
            logger.info(f"Chat history DB: {CHAT_DB_PATH}")

    def get_chat_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Get or create a ChatMessageHistory for a session.

        Uses SQLChatMessageHistory for persistence, or in-memory ChatMessageHistory
        if persistence is disabled.

        Args:
            session_id: Unique session identifier

        Returns:
            ChatMessageHistory instance
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = {}

        if "chat_history" not in self._sessions[session_id]:
            if self.use_persistence:
                try:
                    history = SQLChatMessageHistory(
                        session_id=session_id,
                        connection_string=self.connection_string,
                    )
                    logger.info(f"Loaded persistent history for session: {session_id}")
                except Exception as e:
                    logger.warning(f"Failed to create SQL history: {e}. Using in-memory.")
                    history = ChatMessageHistory()
            else:
                history = ChatMessageHistory()

            self._sessions[session_id]["chat_history"] = history

        return self._sessions[session_id]["chat_history"]

    def get_memory(self, session_id: str, llm=None) -> ConversationBufferWindowMemory:
        """
        Get or create a ConversationBufferWindowMemory for a session.

        This provides a sliding window of the last K conversation exchanges,
        which is passed to the LLM as context for generating responses.

        Args:
            session_id: Unique session identifier
            llm: Optional LLM instance (needed for summary memory)

        Returns:
            ConversationBufferWindowMemory instance
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = {}

        if "memory" not in self._sessions[session_id]:
            chat_history = self.get_chat_history(session_id)

            memory = ConversationBufferWindowMemory(
                chat_memory=chat_history,
                k=MEMORY_WINDOW_SIZE,
                return_messages=True,
                memory_key="chat_history",
                input_key="input",
                output_key="output",
            )

            self._sessions[session_id]["memory"] = memory
            logger.info(
                f"Created buffer window memory (k={MEMORY_WINDOW_SIZE}) "
                f"for session: {session_id}"
            )

        return self._sessions[session_id]["memory"]

    def get_summary_memory(self, session_id: str, llm) -> ConversationSummaryBufferMemory:
        """
        Get or create a ConversationSummaryBufferMemory for a session.

        This combines a buffer of recent messages with a running summary
        of older messages, providing both detail and long-term context.

        Args:
            session_id: Unique session identifier
            llm: LLM instance for generating summaries

        Returns:
            ConversationSummaryBufferMemory instance
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = {}

        if "summary_memory" not in self._sessions[session_id]:
            chat_history = self.get_chat_history(session_id)

            memory = ConversationSummaryBufferMemory(
                llm=llm,
                chat_memory=chat_history,
                max_token_limit=MAX_TOKEN_LIMIT,
                return_messages=True,
                memory_key="chat_history",
                input_key="input",
                output_key="output",
            )

            self._sessions[session_id]["summary_memory"] = memory
            logger.info(
                f"Created summary buffer memory (max_tokens={MAX_TOKEN_LIMIT}) "
                f"for session: {session_id}"
            )

        return self._sessions[session_id]["summary_memory"]

    def get_message_count(self, session_id: str) -> int:
        """Get the number of messages in a session."""
        history = self.get_chat_history(session_id)
        return len(history.messages)

    def get_all_messages(self, session_id: str) -> list[dict]:
        """
        Get all messages for a session as a list of dicts.

        Returns:
            List of {"role": "human"|"ai", "content": "..."} dicts
        """
        history = self.get_chat_history(session_id)
        messages = []
        for msg in history.messages:
            role = "human" if msg.type == "human" else "ai"
            messages.append({
                "role": role,
                "content": msg.content,
                "type": msg.type,
            })
        return messages

    def clear_session(self, session_id: str):
        """Clear all memory for a session."""
        if session_id in self._sessions:
            history = self.get_chat_history(session_id)
            history.clear()
            self._sessions.pop(session_id, None)
            logger.info(f"Cleared memory for session: {session_id}")

    def get_context_summary(self, session_id: str) -> str:
        """
        Get a human-readable summary of the current conversation context.

        Returns:
            String summary of conversation context
        """
        messages = self.get_all_messages(session_id)
        if not messages:
            return "No conversation history yet."

        human_msgs = [m for m in messages if m["role"] == "human"]
        ai_msgs = [m for m in messages if m["role"] == "ai"]

        summary_parts = [
            f"**Messages:** {len(messages)} total ({len(human_msgs)} from user, {len(ai_msgs)} from AI)",
        ]

        # Last topic
        if human_msgs:
            last_msg = human_msgs[-1]["content"]
            summary_parts.append(
                f"**Last topic:** {last_msg[:100]}{'...' if len(last_msg) > 100 else ''}"
            )

        # Key topics from user messages
        all_text = " ".join(m["content"] for m in human_msgs[-5:])
        keywords = self._extract_keywords(all_text)
        if keywords:
            summary_parts.append(f"**Active topics:** {', '.join(keywords[:6])}")

        return "\n\n".join(summary_parts)

    def _extract_keywords(self, text: str, max_keywords: int = 6) -> list[str]:
        """Extract important keywords from text (simple TF-based)."""
        import re
        stop_words = {
            "i", "me", "my", "you", "your", "we", "the", "a", "an", "is", "are",
            "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
            "did", "will", "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into",
            "about", "than", "but", "or", "and", "not", "no", "if", "so", "it", "its",
            "this", "that", "what", "which", "who", "how", "when", "where", "why",
            "am", "just", "also", "very", "much", "more", "most", "some", "any",
            "all", "up", "out", "there", "here", "then", "now", "well", "too",
            "tell", "know", "think", "want", "like", "get", "got", "really",
        }

        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        filtered = [w for w in words if w not in stop_words]

        # Count frequency
        freq = {}
        for w in filtered:
            freq[w] = freq.get(w, 0) + 1

        # Sort by frequency, return top
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:max_keywords]]

    def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())


# Singleton instance
memory_manager = MemoryManager(use_persistence=True)
