export function getOrCreateSessionId(preferredSessionId?: string | undefined): string {
  const key = "infinite-edge-agent.sessionId";
  const preferred = preferredSessionId?.trim();
  if (preferred) {
    localStorage.setItem(key, preferred);
    return preferred;
  }
  const existing = localStorage.getItem(key);
  if (existing) return existing;
  const id = typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `session_${Date.now()}_${Math.random().toString(16).slice(2)}`;
  localStorage.setItem(key, id);
  return id;
}

export function makeMessageId(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `message_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}
