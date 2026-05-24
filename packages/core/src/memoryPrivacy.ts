export type SensitiveFindingKind =
  | "openai_api_key"
  | "github_token"
  | "aws_access_key"
  | "generic_secret";

export interface SensitiveMemoryFinding {
  kind: SensitiveFindingKind;
  replacement: string;
}

export interface RedactedMemoryText {
  text: string;
  findings: SensitiveMemoryFinding[];
}

const SECRET_PATTERNS: Array<{ kind: SensitiveFindingKind; pattern: RegExp }> = [
  { kind: "openai_api_key", pattern: /\bsk-[A-Za-z0-9_-]{20,}\b/g },
  { kind: "github_token", pattern: /\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}\b/g },
  { kind: "github_token", pattern: /\bgithub_pat_[A-Za-z0-9_]{20,}\b/g },
  { kind: "aws_access_key", pattern: /\bAKIA[0-9A-Z]{16}\b/g },
  {
    kind: "generic_secret",
    pattern: /\b(?:api[_-]?key|secret|token|password)\s*[:=]\s*["']?[^"'\s]{12,}["']?/gi
  },
];

export function redactSensitiveMemoryText(text: string): RedactedMemoryText {
  const findings: SensitiveMemoryFinding[] = [];
  let redacted = text;

  for (const { kind, pattern } of SECRET_PATTERNS) {
    redacted = redacted.replace(pattern, () => {
      const replacement = `[redacted:${kind}]`;
      findings.push({ kind, replacement });
      return replacement;
    });
  }

  return { text: redacted, findings };
}
