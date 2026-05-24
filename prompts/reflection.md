# Reflection prompt

After an important assistant response, extract durable memory candidates.

Return only JSON:

```json
{
  "shouldRemember": true,
  "items": [
    {
      "text": "Self-contained memory fact",
      "tags": ["project", "preference"],
      "confidence": 0.9
    }
  ]
}
```

Do not store secrets, credentials, private keys, or short-lived facts unless explicitly requested.
