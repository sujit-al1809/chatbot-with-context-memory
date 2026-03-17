"""
Chains Module — LangChain conversation chains for the chatbot.

Implements:
1. ConversationChain — Main chat chain with memory
2. Custom prompt templates with system prompt and memory context
3. Sentiment and intent analysis utilities
"""
import re
import logging
from typing import Optional

from langchain.chains import ConversationChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser

from chatbot.config import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def create_conversation_chain(llm, memory) -> ConversationChain:
    """
    Create a LangChain ConversationChain with memory.

    This is the core chain that:
    1. Accepts user input
    2. Loads conversation history from memory
    3. Formats the prompt with system instructions + history + user input
    4. Sends to the Mistral LLM
    5. Saves the exchange to memory
    6. Returns the response

    Args:
        llm: LangChain LLM instance
        memory: ConversationBufferWindowMemory instance

    Returns:
        ConversationChain instance
    """
    # Build the prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ])

    chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=False,
        input_key="input",
        output_key="output",
    )

    logger.info("Created conversation chain with memory")
    return chain


def get_response(chain: ConversationChain, user_input: str) -> str:
    """
    Get a response from the conversation chain.

    Args:
        chain: ConversationChain instance
        user_input: User's message

    Returns:
        Assistant's response string
    """
    try:
        result = chain.invoke({"input": user_input})
        response = result.get("output", result.get("response", ""))

        # Clean up response (remove potential prompt artifacts)
        response = _clean_response(response)

        return response

    except Exception as e:
        logger.error(f"Error getting response: {e}")
        return f"I apologize, but I encountered an issue. Error: {str(e)[:100]}"


def _clean_response(text: str) -> str:
    """Clean up LLM response text."""
    # Remove common LLM artifacts
    text = text.strip()

    # Remove repeated newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove potential system prompt leakage
    if "You are ContextBot AI" in text:
        parts = text.split("You are ContextBot AI")
        text = parts[-1].strip()

    return text


# ─── Sentiment Analysis ─────────────────────────────────────

def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment of text using TextBlob with keyword fallback.

    Returns:
        dict with score (-1 to 1), label, and emoji
    """
    try:
        from textblob import TextBlob
        blob = TextBlob(text)
        score = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
    except Exception:
        score = _keyword_sentiment(text)
        subjectivity = 0.5

    # Map to label and emoji
    if score > 0.3:
        label, emoji = ("Very Positive", ":D") if score > 0.6 else ("Positive", ":)")
    elif score < -0.3:
        label, emoji = ("Very Negative", ":'(") if score < -0.6 else ("Negative", ":(")
    else:
        label, emoji = "Neutral", ":|"

    return {
        "score": round(score, 3),
        "label": label,
        "emoji": emoji,
        "subjectivity": round(subjectivity, 3),
    }


def _keyword_sentiment(text: str) -> float:
    """Fallback keyword-based sentiment scoring."""
    positive = {"good", "great", "awesome", "excellent", "happy", "love", "wonderful",
                "fantastic", "amazing", "perfect", "thanks", "thank", "beautiful",
                "brilliant", "best", "nice", "cool", "superb", "enjoy", "helpful"}
    negative = {"bad", "terrible", "awful", "horrible", "hate", "worst", "annoying",
                "frustrated", "angry", "sad", "disappointed", "poor", "wrong",
                "fail", "broken", "ugly", "boring", "stupid", "useless", "confused"}

    words = set(re.findall(r'\w+', text.lower()))
    pos = len(words & positive)
    neg = len(words & negative)
    total = pos + neg
    return (pos - neg) / total if total > 0 else 0.0


# ─── Intent Detection ────────────────────────────────────────

INTENT_PATTERNS = {
    "greeting": (r"\b(hi|hello|hey|howdy|greetings|good\s*(morning|afternoon|evening))\b",
                 {"hi", "hello", "hey", "howdy", "greetings"}),
    "farewell": (r"\b(bye|goodbye|see\s*you|take\s*care|good\s*night)\b",
                 {"bye", "goodbye", "farewell", "later"}),
    "question": (r"(\?$|\b(what|who|where|when|why|how|which|can\s+you|do\s+you)\b)",
                 {"what", "who", "where", "when", "why", "how"}),
    "help": (r"\b(help|assist|support|stuck|confused|explain)\b",
             {"help", "assist", "stuck", "confused", "explain"}),
    "gratitude": (r"\b(thanks|thank|appreciate|grateful)\b",
                  {"thanks", "thank", "appreciate"}),
    "command": (r"\b(show|tell|give|find|search|list|create|set|change|update)\b",
                {"show", "tell", "find", "search", "create", "update"}),
    "opinion": (r"\b(think|feel|believe|opinion|suggest|recommend)\b",
                {"think", "feel", "opinion", "suggest", "recommend"}),
}


def detect_intent(text: str) -> dict:
    """
    Detect the intent of a user message.

    Returns:
        dict with intent name and confidence
    """
    text_lower = text.lower().strip()
    words = set(re.findall(r'\w+', text_lower))
    scores = {}

    for intent, (pattern, keywords) in INTENT_PATTERNS.items():
        score = 0.0
        if re.search(pattern, text_lower, re.IGNORECASE):
            score += 0.5
        overlap = words & keywords
        if overlap:
            score += 0.3 * len(overlap)
        if score > 0:
            scores[intent] = min(score, 1.0)

    if not scores:
        return {"intent": "general", "confidence": 0.3}

    best = max(scores, key=scores.get)
    return {"intent": best, "confidence": round(scores[best], 3)}
