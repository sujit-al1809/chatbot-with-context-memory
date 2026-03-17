"""
Configuration for the Conversational AI Chatbot.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── LLM Configuration ──────────────────────────────────
# OpenRouter (primary — cloud, no GPU needed)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"



# HuggingFace (cloud fallback)
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

# ─── Memory Configuration ───────────────────────────────
MEMORY_WINDOW_SIZE = int(os.getenv("MEMORY_WINDOW_SIZE", "10"))
MAX_TOKEN_LIMIT = int(os.getenv("MAX_TOKEN_LIMIT", "2048"))

# ─── App Configuration ──────────────────────────────────
APP_TITLE = os.getenv("APP_TITLE", "ContextBot AI")

# ─── System Prompt ───────────────────────────────────────
SYSTEM_PROMPT = """You are ContextBot AI, an intelligent and friendly conversational assistant with contextual memory.

Your capabilities:
- You remember the entire conversation history and use it to provide contextual responses.
- You detect the user's intent and sentiment to tailor your tone.
- You recall facts the user has shared (name, preferences, interests) and reference them naturally.
- You provide helpful, accurate, and engaging responses.

Guidelines:
- Be warm, professional, and helpful.
- If the user shares personal details (name, interests, job), acknowledge and remember them.
- Reference previous parts of the conversation when relevant.
- Keep responses concise but informative.
- Use markdown formatting when helpful (bold, lists, code blocks).
"""

# ─── Database ────────────────────────────────────────────
CHAT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chat_history.db")
