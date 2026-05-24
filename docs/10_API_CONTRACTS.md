# API Contracts

## Memory upsert

```http
POST /memory/upsert
Content-Type: application/json
```

Request:

```json
{
  "chunks": [
    {
      "id": "chunk-id",
      "text": "Remembered text",
      "embedding": [0.1, 0.2],
      "sessionId": "session-id",
      "source": "chat",
      "role": "user",
      "createdAt": "2026-05-05T00:00:00.000Z",
      "updatedAt": "2026-05-05T00:00:00.000Z",
      "tags": ["user"],
      "metadata": {},
      "tokenCount": 12
    }
  ]
}
```

Response:

```json
{
  "ok": true,
  "count": 1
}
```

## Memory search

```http
POST /memory/search
Content-Type: application/json
```

Request:

```json
{
  "embedding": [0.1, 0.2],
  "options": {
    "limit": 8,
    "minScore": 0.15,
    "sessionId": "optional-session-id",
    "tags": ["project"],
    "maxAgeMs": 2592000000
  }
}
```

Response:

```json
{
  "hits": [
    {
      "id": "chunk-id",
      "text": "Remembered text",
      "embedding": [0.1, 0.2],
      "sessionId": "session-id",
      "source": "chat",
      "role": "user",
      "createdAt": "2026-05-05T00:00:00.000Z",
      "updatedAt": "2026-05-05T00:00:00.000Z",
      "tags": ["user"],
      "metadata": {},
      "tokenCount": 12,
      "score": 0.91
    }
  ]
}
```

## Memory clear

```http
DELETE /memory
```

Response:

```json
{
  "ok": true
}
```
