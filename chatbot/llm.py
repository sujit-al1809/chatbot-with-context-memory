"""
LLM Module - Sets up the Mistral-7B language model via OpenRouter or HuggingFace.

Supports:
1. OpenRouter (recommended) — Cloud API, access Mistral-7B for free, no GPU needed
2. HuggingFace Inference API — Cloud-based fallback
3. Fake LLM — For testing/demo when no LLM is available
"""
import logging
from langchain_core.language_models.llms import BaseLLM

from chatbot.config import (
    OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_BASE_URL,
    HF_API_TOKEN, HF_MODEL,
)

logger = logging.getLogger(__name__)


def get_llm(provider: str = "openrouter", temperature: float = 0.7):
    """
    Initialize and return the LLM based on provider.

    Args:
        provider: "openrouter", "huggingface", or "fake"
        temperature: Controls response randomness (0.0 = deterministic, 1.0 = creative)

    Returns:
        LangChain LLM instance
    """
    if provider == "openrouter":
        return _get_openrouter_llm(temperature)
    elif provider == "huggingface":
        return _get_huggingface_llm(temperature)
    elif provider == "fake":
        return _get_fake_llm()
    else:
        logger.warning(f"Unknown provider '{provider}', falling back to OpenRouter")
        return _get_openrouter_llm(temperature)


def _get_openrouter_llm(temperature: float):
    """
    Initialize Mistral-7B via OpenRouter (free tier available).

    OpenRouter provides access to many LLMs via a single API.
    Free models include mistral-7b-instruct.

    Prerequisites:
    1. Sign up at https://openrouter.ai
    2. Get your API key from https://openrouter.ai/keys
    3. Set OPENROUTER_API_KEY in your .env file
    """
    try:
        from langchain_community.chat_models import ChatOpenAI

        if not OPENROUTER_API_KEY:
            logger.warning("No OpenRouter API key found. Set OPENROUTER_API_KEY in .env")
            logger.info("Falling back to fake LLM for demo mode")
            return _get_fake_llm()

        llm = ChatOpenAI(
            model=OPENROUTER_MODEL,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base=OPENROUTER_BASE_URL,
            temperature=temperature,
            max_tokens=512,
            default_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "ContextBot AI",
            },
        )
        logger.info(f"OpenRouter LLM initialized: model={OPENROUTER_MODEL}")
        return llm

    except Exception as e:
        logger.error(f"Failed to initialize OpenRouter: {e}")
        logger.info("Falling back to fake LLM for demo mode")
        return _get_fake_llm()





def _get_huggingface_llm(temperature: float):
    """
    Initialize Mistral-7B via HuggingFace Inference API.

    Prerequisites:
    1. Get API token from https://huggingface.co/settings/tokens
    2. Set HUGGINGFACEHUB_API_TOKEN in .env
    """
    try:
        from langchain_community.llms import HuggingFaceHub

        if not HF_API_TOKEN:
            logger.warning("No HuggingFace API token found. Falling back to fake LLM.")
            return _get_fake_llm()

        llm = HuggingFaceHub(
            repo_id=HF_MODEL,
            huggingfacehub_api_token=HF_API_TOKEN,
            model_kwargs={
                "temperature": temperature,
                "max_new_tokens": 512,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
            }
        )
        logger.info(f"HuggingFace LLM initialized: model={HF_MODEL}")
        return llm

    except Exception as e:
        logger.error(f"Failed to initialize HuggingFace LLM: {e}")
        return _get_fake_llm()


def _get_fake_llm():
    """
    Fake LLM for testing/demo when no real LLM is available.
    Uses rule-based responses.
    """
    from langchain_community.llms.fake import FakeListLLM

    responses = [
        "That's a great question! Based on our conversation, I think the answer lies in understanding the context better. Could you tell me more about what you're looking for?",
        "I appreciate you sharing that! I've noted it in my memory. Is there anything specific you'd like to explore further?",
        "Interesting point! From what we've discussed so far, I can see a pattern emerging. Let me help you think through this.",
        "Thanks for the context! I remember you mentioned something related earlier. Let me connect the dots for you.",
        "I understand your concern. Based on our conversation history, here's what I think would work best for you.",
        "Great observation! I'll keep that in mind for our future conversations. What else would you like to discuss?",
        "That's a fascinating topic! I've stored this in my contextual memory. Feel free to ask me about it anytime.",
        "I hear you! Let me use the context from our chat to give you a more tailored response.",
    ]

    logger.info("Using Fake LLM (demo mode). Set OPENROUTER_API_KEY in .env for real responses.")
    return FakeListLLM(responses=responses)





def check_openrouter_connection() -> dict:
    """
    Check if OpenRouter API key is configured and working.

    Returns:
        dict with status info
    """
    if not OPENROUTER_API_KEY:
        return {
            "status": "no_key",
            "message": "No API key set. Get a free key at https://openrouter.ai/keys"
        }

    try:
        import requests
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            timeout=5
        )
        if response.status_code == 200:
            return {
                "status": "connected",
                "message": f"OpenRouter connected - Using {OPENROUTER_MODEL}"
            }
        return {
            "status": "error",
            "message": f"OpenRouter returned status {response.status_code}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Cannot reach OpenRouter: {str(e)[:80]}"
        }
