# Conversational AI Chatbot with Contextual Memory

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Mistral](https://img.shields.io/badge/Mistral--7B-LLM-FF7000?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

**An AI-powered QA chatbot using LangChain and Streamlit integrated with the Mistral LLM for context-aware conversations. Implements conversational memory using LangChain ChatMessageHistory to retain dialogue context and improve response continuity.**

[Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [Tech Stack](#-tech-stack) · [Demo](#-demo-guide)

</div>

---

## Features

### Context-Aware Conversations
- **LangChain ConversationChain** with Mistral-7B for intelligent, contextual responses
- **Sliding context window** using `ConversationBufferWindowMemory` (retains last K exchanges)
- **Message continuity** - the LLM receives full conversation history for coherent multi-turn dialogue

### Conversational Memory (LangChain ChatMessageHistory)
- **`ChatMessageHistory`** — Per-session conversation storage via LangChain
- **`SQLChatMessageHistory`** — SQLite-backed persistence across app restarts
- **`ConversationSummaryBufferMemory`** — Auto-summarizes older messages while keeping recent ones detailed
- **Entity extraction** — Automatically captures user name, location, occupation, and preferences

### Mistral-7B Integration
- **OpenRouter** - Access Mistral-7B via Cloud API for free
- **HuggingFace Inference API** - Cloud fallback using `mistralai/Mistral-7B-Instruct-v0.2`
- **Demo mode** - Fake LLM for testing without model setup
- **Configurable** - Adjust temperature, context window, and model parameters

### Intent Detection
- Hybrid pattern + keyword matching across 7 intent categories
- Real-time badges showing detected intent with confidence score
- Categories: greeting, farewell, question, help, gratitude, command, opinion

### Sentiment Analysis
- **TextBlob** polarity scoring with keyword-based fallback
- 5 sentiment labels with emoji indicators
- Real-time sentiment tracking per message
- Session-level sentiment distribution analytics

### Analytics Dashboard
- Total messages and sessions counter
- Sentiment distribution with progress bars
- Intent frequency breakdown
- All accessible from the sidebar

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   Streamlit UI (app.py)                       │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ Chat UI  │  │ Sidebar      │  │ Analytics Dashboard   │  │
│  │ st.chat_ │  │ • Sessions   │  │ • Sentiment dist.     │  │
│  │ message  │  │ • Memory     │  │ • Intent breakdown    │  │
│  │ st.chat_ │  │ • LLM config │  │ • Message stats       │  │
│  │ input    │  │ • Entities   │  │                       │  │
│  └────┬─────┘  └──────────────┘  └───────────────────────┘  │
│       │                                                      │
└───────┼──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│                   LangChain Pipeline                         │
│                                                              │
│  User Input                                                  │
│      │                                                       │
│      ├──▶ NLP Analysis (sentiment, intent, entities)         │
│      │                                                       │
│      ▼                                                       │
│  ┌──────────────────────────────────────┐                    │
│  │     ConversationChain                │                    │
│  │  ┌────────────────────────────────┐  │                    │
│  │  │ ChatPromptTemplate             │  │                    │
│  │  │  ├─ SystemMessage (persona)    │  │                    │
│  │  │  ├─ MessagesPlaceholder        │◀─┼── ChatMessageHistory
│  │  │  │   (chat_history)            │  │    (SQLite backed) │
│  │  │  └─ HumanMessage (user input)  │  │                    │
│  │  └────────────────────────────────┘  │                    │
│  │              │                       │                    │
│  │              ▼                       │                    │
│  │  ┌────────────────────────────────┐  │                    │
│  │  │ Mistral-7B LLM (via OpenRouter)│  │                    │
│  │  └────────────────────────────────┘  │                    │
│  │              │                       │                    │
│  │              ▼                       │                    │
│  │  ConversationBufferWindowMemory      │                    │
│  │  (saves exchange, sliding window)    │                    │
│  └──────────────────────────────────────┘                    │
│              │                                               │
│              ▼                                               │
│         Response → Display in Streamlit                      │
└──────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites
- **Python 3.10+**

### 1. Clone & Setup Virtual Environment

```bash
cd contextual-memory-project

# Create virtual environment
python -m venv venv

# Activate venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

The app opens at **[http://localhost:8501](http://localhost:8501)**

### 3. Run Tests

```bash
pytest tests/ -v
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend + UI** | Streamlit | Chat interface, sidebar, analytics |
| **LLM Framework** | LangChain | Chains, memory, prompt templates |
| **Language Model** | Mistral-7B | Response generation |
| **Memory** | LangChain ChatMessageHistory | Conversation context retention |
| **Persistence** | SQLite (SQLChatMessageHistory) | Chat history across sessions |
| **Sentiment** | TextBlob | Polarity & subjectivity scoring |
| **Intent** | Custom (regex + keywords) | User intent classification |
| **Testing** | pytest | Unit tests |

---

## Project Structure

```
contextual-memory-project/
├── app.py                      # Main Streamlit application (UI + chat logic)
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
│
├── chatbot/                    # Core chatbot module
│   ├── __init__.py
│   ├── config.py               # Configuration (LLM, memory, prompts)
│   ├── llm.py                  # LLM setup (OpenRouter/HuggingFace/Fake)
│   ├── memory.py               # Conversational memory (ChatMessageHistory)
│   ├── chains.py               # LangChain ConversationChain + NLP utilities
│   └── utils.py                # Helper functions + entity extraction
│
├── data/
│   ├── intents.json            # Intent patterns & response templates
│   ├── responses.json          # Fallback response templates
│   └── chat_history.db         # SQLite DB (auto-created at runtime)
│
├── tests/
│   ├── __init__.py
│   ├── test_chains.py          # Tests for chains, sentiment, intent
│   ├── test_memory_service.py  # Tests for memory manager
│   └── test_llm.py             # Tests for LLM + utilities
│
└── docs/
    ├── API_DOCUMENTATION.md    # Internal API reference
    └── ARCHITECTURE.md         # Architecture deep dive
```

---

## Demo Guide

### 1. Start the app
```bash
streamlit run app.py
```

### 2. Test Memory & Context
```
You: "My name is Sujit and I am a software engineer"
You: "I love working with Python and machine learning"
You: "What do you know about me?"
-> Bot should recall your name, job, and interests!
```

### 3. Test Sentiment Detection
```
You: "I'm so happy and excited about this project!" -> Positive
You: "This is frustrating, nothing works"             -> Negative
You: "The meeting is at 3 PM"                          -> Neutral
```

### 4. Test Intent Detection
```
You: "Hello there!"              -> greeting
You: "What is deep learning?"    -> question
You: "Help me understand LLMs"   -> help
You: "Thank you so much!"        -> gratitude
```

### 5. Check Analytics
Open the **Analytics** section in the sidebar to see real-time stats.

---

## Key Technical Highlights (For Placement Interviews)

### 1. **LangChain ConversationChain**
> The chain integrates `ChatPromptTemplate` -> `MessagesPlaceholder` (for memory) -> Mistral LLM. Memory is injected via `ConversationBufferWindowMemory` which slides over the last K exchanges.

### 2. **ChatMessageHistory (LangChain)**
> We use `SQLChatMessageHistory` backed by SQLite for persistence. Each session gets its own history instance, enabling multi-session support with independent memory.

### 3. **Mistral-7B via OpenRouter**
> OpenRouter provides a serverless API to access the Mistral model. We connect via LangChain with configurable temperature and context window.

### 4. **Graceful Degradation**
> If the API isn't running, the system falls back to a `FakeListLLM` for demo purposes, ensuring the app always works. Similarly, TextBlob sentiment falls back to keyword-based scoring.

### 5. **Entity Extraction**
> Regex-based extraction captures user name, location, occupation, and preferences from natural language - these are stored and displayed in the Memory panel.

---

## License

Built for educational and placement purposes.

---

<div align="center">

**Built with Python · LangChain · Mistral-7B · Streamlit**

</div>
