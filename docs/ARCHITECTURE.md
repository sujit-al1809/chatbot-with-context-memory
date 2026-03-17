# Architecture Deep Dive

## System Overview

The Conversational AI Chatbot is built with **LangChain** as the orchestration framework, **Mistral-7B** (via OpenRouter) as the language model, and **Streamlit** as the full-stack UI.

---

## Core Components

### 1. LangChain ConversationChain

The heart of the chatbot. The chain processes each user message through:

```python
ConversationChain(
    llm=ChatOpenAI(model="mistralai/mistral-7b-instruct:free"), # Mistral-7B LLM
    memory=ConversationBufferWindowMemory(  # Sliding window memory
        chat_memory=SQLChatMessageHistory(  # SQLite persistence
            session_id="..."
        ),
        k=10                                # Last 10 exchanges
    ),
    prompt=ChatPromptTemplate([             # Structured prompt
        SystemMessage(SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        HumanMessage("{input}")
    ])
)
```

### 2. Memory Architecture

```
                    LangChain Memory Stack
                    ══════════════════════

┌─────────────────────────────────────────────┐
│         ConversationBufferWindowMemory       │
│         (Sliding window of K exchanges)      │
│                                              │
│    ┌──────────────────────────────────────┐  │
│    │      SQLChatMessageHistory           │  │
│    │      (SQLite persistence)            │  │
│    │                                      │  │
│    │  Session A: [msg1, msg2, ..., msgN]  │  │
│    │  Session B: [msg1, msg2, ..., msgM]  │  │
│    │  Session C: [msg1, msg2, ...]        │  │
│    └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

**Memory types used:**
- `ChatMessageHistory` — In-memory per-session history
- `SQLChatMessageHistory` — SQLite-backed persistent storage
- `ConversationBufferWindowMemory` — Sliding window of last K exchanges
- `ConversationSummaryBufferMemory` — Auto-summarization for long conversations

### 3. NLP Pipeline

Every user message is analyzed before being sent to the LLM:

```
User Message
    │
    ├──> Sentiment Analysis (TextBlob polarity)
    │    -> score (-1 to 1), label
    │
    ├──> Intent Detection (regex + keywords)
    │    -> intent category + confidence score
    │
    ├──> Entity Extraction (regex patterns)
    │    -> name, location, occupation, preferences
    │
    └──> ConversationChain.invoke()
         -> Mistral-7B generates context-aware response
```

### 4. LLM Provider Strategy

```
get_llm(provider)
    │
    ├── "openrouter"   -> ChatOpenAI(model="mistralai/mistral-7b-instruct:free")
    ├── "huggingface"  -> HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2")
    └── "fake"         -> FakeListLLM(responses=[...])   # Demo mode
```

---

## Data Flow

1. **User types message** -> Streamlit captures via `st.chat_input()`
2. **NLP analysis** -> Sentiment, intent, entities extracted
3. **Chain invoked** -> `ConversationChain.invoke({"input": message})`
4. **Memory loads** -> Last K exchanges loaded from `ChatMessageHistory`
5. **Prompt built** -> System prompt + history + user input formatted
6. **Mistral generates** -> Response via OpenRouter API
7. **Memory saves** -> Exchange stored in `SQLChatMessageHistory`
8. **Display** -> Response + NLP badges rendered in Streamlit

---

## Database Schema (SQLite)

The `SQLChatMessageHistory` automatically creates:

```sql
CREATE TABLE message_store (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    message TEXT NOT NULL,     -- JSON: {"type": "human/ai", "content": "..."}
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## Testing Strategy

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_chains.py` | 15 | Sentiment, intent, chain, response cleaning |
| `test_memory_service.py` | 10 | ChatMessageHistory, sessions, context |
| `test_llm.py` | 6 | LLM init, entity extraction, utilities |
