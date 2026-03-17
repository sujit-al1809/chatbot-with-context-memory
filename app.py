"""
Conversational AI Chatbot with Contextual Memory
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Main Streamlit Application

Tech Stack: Python, LangChain, Mistral-7B (OpenRouter), Streamlit
Features:  Contextual memory, sentiment analysis, intent detection,
           session management, analytics dashboard
"""
import streamlit as st
import uuid
import json
import time
from datetime import datetime

from chatbot.llm import get_llm, check_openrouter_connection
from chatbot.memory import memory_manager
from chatbot.chains import (
    create_conversation_chain,
    get_response,
    analyze_sentiment,
    detect_intent,
)
from chatbot.utils import generate_session_id, format_timestamp, extract_entities


# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ContextBot AI - Conversational AI with Memory",
    page_icon="App",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS — Premium Dark Theme
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ── */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 0.9rem;
    }

    /* ── Chat Messages ── */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
    }

    .user-msg {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
        padding: 14px 20px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        font-size: 0.95rem;
        line-height: 1.6;
        max-width: 85%;
        float: right;
        clear: both;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2);
    }

    .bot-msg {
        background: rgba(30, 30, 50, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        color: #e2e8f0;
        padding: 14px 20px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        font-size: 0.95rem;
        line-height: 1.6;
        max-width: 85%;
        float: left;
        clear: both;
        backdrop-filter: blur(10px);
    }

    /* ── Badges ── */
    .badge-row {
        display: flex;
        gap: 6px;
        margin-top: 6px;
        flex-wrap: wrap;
        clear: both;
    }
    .badge {
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-intent {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.25);
    }
    .badge-sentiment-positive {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.25);
    }
    .badge-sentiment-negative {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.25);
    }
    .badge-sentiment-neutral {
        background: rgba(148, 163, 184, 0.15);
        color: #94a3b8;
        border: 1px solid rgba(148, 163, 184, 0.25);
    }

    /* ── Sidebar ── */
    .sidebar-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.1);
        margin-bottom: 1rem;
    }
    .sidebar-header h2 {
        font-size: 1.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .sidebar-header p {
        color: #64748b;
        font-size: 0.75rem;
    }

    /* ── Stats Cards ── */
    .stat-card {
        background: rgba(30, 30, 50, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin-bottom: 8px;
    }
    .stat-card .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stat-card .stat-label {
        color: #64748b;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Memory Card ── */
    .memory-card {
        background: rgba(30, 30, 50, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
    .memory-card .memory-type {
        color: #a78bfa;
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .memory-card .memory-content {
        color: #cbd5e1;
        margin-top: 4px;
        line-height: 1.4;
    }

    /* ── Status Indicator ── */
    .status-connected {
        color: #4ade80;
        font-size: 0.8rem;
    }
    .status-disconnected {
        color: #f87171;
        font-size: 0.8rem;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Welcome Card ── */
    .welcome-card {
        background: rgba(30, 30, 50, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }
    .welcome-card h3 {
        margin-bottom: 6px;
    }
    .welcome-card p {
        color: #94a3b8;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════

def init_session_state():
    """Initialize all session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = generate_session_id()

    if "sessions" not in st.session_state:
        st.session_state.sessions = {
            st.session_state.session_id: {
                "title": "New Conversation",
                "created_at": datetime.now(),
                "messages": [],          # [{role, content, timestamp, analysis}]
                "entities": {},          # Extracted entities
            }
        }

    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = "openrouter"

    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7

    if "llm" not in st.session_state:
        st.session_state.llm = None

    if "chain" not in st.session_state:
        st.session_state.chain = None

    if "analytics" not in st.session_state:
        st.session_state.analytics = {
            "total_messages": 0,
            "sentiments": [],
            "intents": [],
        }


init_session_state()


# ═══════════════════════════════════════════════════════════════
# LLM INITIALIZATION
# ═══════════════════════════════════════════════════════════════

def init_llm():
    """Initialize or reinitialize the LLM and chain."""
    provider = st.session_state.llm_provider
    temperature = st.session_state.temperature

    llm = get_llm(provider=provider, temperature=temperature)
    st.session_state.llm = llm

    # Create chain with memory
    session_id = st.session_state.session_id
    memory = memory_manager.get_memory(session_id, llm=llm)
    chain = create_conversation_chain(llm, memory)
    st.session_state.chain = chain


def ensure_chain():
    """Ensure the chain is initialized for the current session."""
    if st.session_state.chain is None:
        init_llm()

    # If session changed, reinitialize chain with new memory
    session_id = st.session_state.session_id
    memory = memory_manager.get_memory(session_id, llm=st.session_state.llm)
    st.session_state.chain = create_conversation_chain(st.session_state.llm, memory)


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    # Header
    st.markdown("""
    <div class="sidebar-header">
        <h2>ContextBot AI</h2>
        <p>LangChain · Mistral-7B · Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("New Conversation", use_container_width=True, type="primary"):
        new_id = generate_session_id()
        st.session_state.session_id = new_id
        st.session_state.sessions[new_id] = {
            "title": "New Conversation",
            "created_at": datetime.now(),
            "messages": [],
            "entities": {},
        }
        st.session_state.chain = None  # Will reinitialize
        st.rerun()

    st.divider()

    # ── Session List ──
    st.markdown("**Sessions**")
    for sid, session in sorted(
        st.session_state.sessions.items(),
        key=lambda x: x[1]["created_at"],
        reverse=True
    ):
        is_active = sid == st.session_state.session_id
        label = f"{'> ' if is_active else ''}{session['title']}"

        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(
                label,
                key=f"session_{sid}",
                use_container_width=True,
                disabled=is_active,
            ):
                st.session_state.session_id = sid
                st.session_state.chain = None
                st.rerun()
        with col2:
            if not is_active and st.button("Del", key=f"del_{sid}"):
                memory_manager.clear_session(sid)
                del st.session_state.sessions[sid]
                st.rerun()

    st.divider()

    # ── LLM Settings ──
    with st.expander("LLM Settings", expanded=False):
        provider = st.selectbox(
            "Provider",
            ["openrouter", "huggingface", "fake"],
            index=["openrouter", "huggingface", "fake"].index(st.session_state.llm_provider),
            help="**openrouter**: Cloud Mistral-7B (free, no GPU)\n**huggingface**: Cloud API\n**fake**: Demo mode"
        )
        if provider != st.session_state.llm_provider:
            st.session_state.llm_provider = provider
            st.session_state.chain = None

        temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.1)
        if temperature != st.session_state.temperature:
            st.session_state.temperature = temperature
            st.session_state.chain = None

        # OpenRouter status
        if provider == "openrouter":
            status = check_openrouter_connection()
            if status["status"] == "connected":
                st.markdown(f'<p class="status-connected">{status["message"]}</p>',
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="status-disconnected">{status["message"]}</p>',
                           unsafe_allow_html=True)
                st.info("💡 Get a free API key at [openrouter.ai/keys](https://openrouter.ai/keys)")


        if st.button("Reinitialize LLM", use_container_width=True):
            st.session_state.chain = None
            st.rerun()

    # ── Memory Panel ──
    with st.expander("Memory & Context", expanded=False):
        sid = st.session_state.session_id
        context = memory_manager.get_context_summary(sid)
        st.markdown(context)

        # Show extracted entities
        current_session = st.session_state.sessions.get(sid, {})
        entities = current_session.get("entities", {})
        if entities:
            st.markdown("**Extracted Entities:**")
            for etype, evalue in entities.items():
                st.markdown(f"""
                <div class="memory-card">
                    <div class="memory-type">{etype}</div>
                    <div class="memory-content">{evalue}</div>
                </div>
                """, unsafe_allow_html=True)

        st.caption(f"{msg_count} messages in LangChain memory")

    # ── Analytics ──
    with st.expander("Analytics", expanded=False):
        analytics = st.session_state.analytics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{analytics['total_messages']}</div>
                <div class="stat-label">Messages</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{len(st.session_state.sessions)}</div>
                <div class="stat-label">Sessions</div>
            </div>
            """, unsafe_allow_html=True)

        # Sentiment breakdown
        sentiments = analytics.get("sentiments", [])
        if sentiments:
            from collections import Counter
            counts = Counter(sentiments)
            st.markdown("**Sentiment Distribution:**")
            for label in ["Positive", "Neutral", "Negative"]:
                count = counts.get(label, 0) + counts.get(f"Very {label}", 0)
                total = len(sentiments) or 1
                pct = count / total
                emoji = {"Positive": ":)", "Neutral": ":|", "Negative": ":("}[label]
                st.progress(pct, text=f"{emoji} {label}: {count}")

        # Intent breakdown
        intents = analytics.get("intents", [])
        if intents:
            from collections import Counter
            intent_counts = Counter(intents)
            st.markdown("**Intent Distribution:**")
            for intent, count in intent_counts.most_common(5):
                st.caption(f"Intent: {intent} ({count})")


# ═══════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ═══════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="main-header">
    <h1>ContextBot AI</h1>
    <p>Conversational AI Chatbot with Contextual Memory - Powered by LangChain & Mistral-7B</p>
</div>
""", unsafe_allow_html=True)

# Get current session
current_session = st.session_state.sessions.get(st.session_state.session_id, {})
messages = current_session.get("messages", [])

# ── Display Chat History ──
if not messages:
    # Welcome screen
    col1, col2, col3, col4 = st.columns(4)
    features = [
        ("Context", "Context Memory", "Remembers conversations using LangChain ChatMessageHistory"),
        ("Intent", "Intent Detection", "Classifies user intent with pattern & keyword matching"),
        ("Sentiment", "Sentiment Analysis", "Tracks mood with TextBlob polarity scoring"),
        ("Mistral-7B", "Mistral-7B LLM", "Powered by Mistral via OpenRouter — no GPU needed"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(f"""
            <div class="welcome-card">
                <h3>{icon}</h3>
                <h3>{title}</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

# ── Render Messages ──
for msg in messages:
    role = msg["role"]
    content = msg["content"]
    analysis = msg.get("analysis", {})

    with st.chat_message(role, avatar="User" if role == "user" else "Bot"):
        st.markdown(content)

        # Show analysis badges for user messages
        if role == "user" and analysis:
            sentiment = analysis.get("sentiment", {})
            intent = analysis.get("intent", {})

            sent_label = sentiment.get("label", "Neutral")
            sent_emoji = sentiment.get("emoji", "")
            sent_class = "positive" if "Positive" in sent_label else (
                "negative" if "Negative" in sent_label else "neutral"
            )
            intent_name = intent.get("intent", "general")
            intent_conf = intent.get("confidence", 0)

            st.markdown(f"""
            <div class="badge-row">
                <span class="badge badge-intent">{intent_name} ({intent_conf:.0%})</span>
                <span class="badge badge-sentiment-{sent_class}">{sent_label}</span>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# CHAT INPUT & PROCESSING
# ═══════════════════════════════════════════════════════════════

if user_input := st.chat_input("Type your message..."):
    # Ensure chain is ready
    ensure_chain()

    # ── 1. Display user message ──
    with st.chat_message("user", avatar="User"):
        st.markdown(user_input)

    # ── 2. NLP Analysis ──
    sentiment = analyze_sentiment(user_input)
    intent = detect_intent(user_input)
    entities = extract_entities(user_input)

    # Show analysis badges
    with st.chat_message("user", avatar="User"):
        sent_label = sentiment["label"]
        sent_class = "positive" if "Positive" in sent_label else (
            "negative" if "Negative" in sent_label else "neutral"
        )
        st.markdown(f"""
        <div class="badge-row">
            <span class="badge badge-intent">{intent['intent']} ({intent['confidence']:.0%})</span>
            <span class="badge badge-sentiment-{sent_class}">{sent_label}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── 3. Store user message ──
    user_msg = {
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().isoformat(),
        "analysis": {
            "sentiment": sentiment,
            "intent": intent,
            "entities": entities,
        }
    }
    messages.append(user_msg)

    # Update entities
    if entities:
        current_session.setdefault("entities", {}).update(entities)

    # Update analytics
    st.session_state.analytics["total_messages"] += 1
    st.session_state.analytics["sentiments"].append(sentiment["label"])
    st.session_state.analytics["intents"].append(intent["intent"])

    # ── 4. Get LLM response ──
    with st.chat_message("assistant", avatar="Bot"):
        with st.spinner("Thinking..."):
            response = get_response(st.session_state.chain, user_input)
        st.markdown(response)

    # ── 5. Store assistant message ──
    bot_msg = {
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now().isoformat(),
    }
    messages.append(bot_msg)
    st.session_state.analytics["total_messages"] += 1

    # ── 6. Auto-title from first message ──
    if len(messages) <= 2:
        title = user_input[:40] + ("..." if len(user_input) > 40 else "")
        current_session["title"] = title

    # Rerun to display properly
    st.rerun()
