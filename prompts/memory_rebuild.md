# Memory rebuild prompt

Use this prompt for a future summarization job that converts raw memory chunks into a compact session summary.

Input:

- Recent messages
- Retrieved memory clusters
- Project files or documents

Output JSON:

```json
{
  "summary": "Compact project state summary",
  "stablePreferences": ["Preference 1"],
  "openTasks": ["Task 1"],
  "decisions": [
    {
      "decision": "What was decided",
      "reason": "Why",
      "date": "ISO timestamp"
    }
  ],
  "risks": ["Risk 1"],
  "nextActions": ["Action 1"]
}
```
