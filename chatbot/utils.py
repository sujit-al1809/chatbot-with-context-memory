"""
Utilities for the chatbot — helper functions and constants.
"""
import uuid
import re
from datetime import datetime


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return str(uuid.uuid4())[:8]


def format_timestamp(dt: datetime = None) -> str:
    """Format a datetime for display."""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%I:%M %p")


def truncate(text: str, max_len: int = 50) -> str:
    """Truncate text with ellipsis."""
    return text[:max_len] + "..." if len(text) > max_len else text


def count_tokens_approx(text: str) -> int:
    """Approximate token count (1 token ≈ 4 chars for English)."""
    return len(text) // 4


def extract_entities(text: str) -> dict:
    """
    Extract named entities from text using regex patterns.

    Returns dict of entity_type -> [values]
    """
    entities = {}

    # Name
    name_match = re.search(
        r"(?:my name is|i'm|i am|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        text, re.IGNORECASE
    )
    if name_match:
        entities["name"] = name_match.group(1).strip()

    # Location
    loc_match = re.search(
        r"(?:i live|i'm from|i am from|i come from|based in)\s+(.+?)(?:\.|,|!|$)",
        text, re.IGNORECASE
    )
    if loc_match:
        entities["location"] = loc_match.group(1).strip()

    # Occupation
    job_match = re.search(
        r"(?:i work as|i am a|i'm a|my job is|i work at)\s+(.+?)(?:\.|,|!|$)",
        text, re.IGNORECASE
    )
    if job_match:
        entities["occupation"] = job_match.group(1).strip()

    # Preferences
    like_match = re.search(
        r"(?:i (?:really\s+)?(?:like|love|enjoy|prefer))\s+(.+?)(?:\.|,|!|$)",
        text, re.IGNORECASE
    )
    if like_match:
        entities["preference"] = like_match.group(1).strip()

    return entities
