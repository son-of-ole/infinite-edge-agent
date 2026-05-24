/**
 * Cheap approximation for planning context budgets without tokenizer access.
 * English text commonly averages ~3.5-4.5 chars/token; 4 is adequate for a UI budget.
 */
export function approximateTokenCount(text: string): number {
  if (!text.trim()) return 0;
  return Math.ceil(text.trim().length / 4);
}

export function clampTextToTokenBudget(text: string, maxTokens: number): string {
  if (maxTokens <= 0) return "";
  const maxChars = maxTokens * 4;
  if (text.length <= maxChars) return text;
  return `${text.slice(0, Math.max(0, maxChars - 20)).trimEnd()}\n...[truncated]`;
}
