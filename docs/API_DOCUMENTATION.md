# API Documentation

## Base URL

```
http://localhost:8000
```

## Authentication

No authentication required (development mode).

---

## Endpoints

### 1. Health Check

```
GET /health
```

**Response:**
```json
{
    "status": "healthy",
    "service": "Contextual Memory Chatbot"
}
```

---

### 2. Chat

#### Send Message

```
POST /api/chat/send
```

**Request Body:**
```json
{
    "session_id": "uuid-string",
    "message": "Hello, how are you?"
}
```

**Response:**
```json
{
    "session_id": "uuid-string",
    "user_message": {
        "id": 1,
        "session_id": "uuid-string",
        "role": "user",
        "content": "Hello, how are you?",
        "timestamp": "2024-01-01T00:00:00",
        "sentiment_score": 0.0,
        "sentiment_label": "neutral",
        "intent": "greeting",
        "intent_confidence": 0.8
    },
    "assistant_message": {
        "id": 2,
        "session_id": "uuid-string",
        "role": "assistant",
        "content": "Hello! How can I help you today?",
        "timestamp": "2024-01-01T00:00:01"
    },
    "context_used": [],
    "memories_retrieved": [],
    "nlp_analysis": {
        "sentiment_score": 0.0,
        "sentiment_label": "neutral",
        "intent": "greeting",
        "intent_confidence": 0.8,
        "keywords": [],
        "active_topics": [],
        "context_summary": ""
    }
}
```

#### Get Chat History

```
GET /api/chat/history/{session_id}?limit=100
```

**Response:**
```json
{
    "session_id": "uuid-string",
    "messages": [...],
    "total": 10
}
```

---

### 3. Sessions

#### Create Session

```
POST /api/sessions/create
```

**Request Body:**
```json
{
    "title": "My Conversation"
}
```

#### List Sessions

```
GET /api/sessions/list?limit=50
```

#### Get Session

```
GET /api/sessions/{session_id}
```

#### Delete Session

```
DELETE /api/sessions/{session_id}
```

#### Update Session Title

```
PUT /api/sessions/{session_id}/title?title=New%20Title
```

---

### 4. Memory

#### Semantic Search

```
GET /api/memory/search?q=machine+learning&limit=5&session_id=optional
```

**Response:**
```json
{
    "query": "machine learning",
    "results": [
        {
            "content": "User discussed topic: machine learning",
            "similarity": 0.89,
            "memory_type": "topic",
            "session_id": "uuid-string"
        }
    ],
    "total": 1
}
```

#### Get Context Window

```
GET /api/memory/context/{session_id}
```

#### Get Session Summary

```
GET /api/memory/summary/{session_id}
```

#### Get All Memories

```
GET /api/memory/all?session_id=optional&limit=20
```

#### Get Memory Stats

```
GET /api/memory/stats
```

---

### 5. Analytics

#### Get Overall Stats

```
GET /api/analytics/stats
```

**Response:**
```json
{
    "total_sessions": 5,
    "total_messages": 42,
    "total_memories": 18,
    "avg_messages_per_session": 8.4,
    "sentiment_distribution": {
        "neutral": 15,
        "positive": 10,
        "negative": 2
    },
    "intent_distribution": {
        "question": 12,
        "greeting": 5,
        "command": 3
    },
    "messages_per_day": {
        "2024-01-01": 10,
        "2024-01-02": 8
    }
}
```

#### Get Sentiment Trends

```
GET /api/analytics/sentiment/{session_id}
```

#### Get Topic Frequency

```
GET /api/analytics/topics
```

#### Get Activity Data

```
GET /api/analytics/activity
```

---

## Error Responses

All error responses follow the format:

```json
{
    "detail": "Error description"
}
```

| Status Code | Description |
|-------------|-------------|
| 400 | Bad request (invalid input) |
| 404 | Resource not found |
| 500 | Internal server error |
