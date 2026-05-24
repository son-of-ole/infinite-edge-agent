import { createCipheriv, createDecipheriv, createHash, randomBytes } from "node:crypto";
import type { LanceStringCodec } from "./lancedbStore.js";

const PREFIX = "enc:v1:";

export function createAesGcmStringCodec(secret: string): LanceStringCodec {
  if (!secret.trim()) {
    throw new Error("MEMORY_ENCRYPTION_KEY must not be empty when memory encryption is enabled.");
  }
  const key = createHash("sha256").update(secret).digest();
  return {
    encodeString(value: string): string {
      if (value.startsWith(PREFIX)) return value;
      const iv = randomBytes(12);
      const cipher = createCipheriv("aes-256-gcm", key, iv);
      const ciphertext = Buffer.concat([cipher.update(value, "utf8"), cipher.final()]);
      const tag = cipher.getAuthTag();
      return `${PREFIX}${toBase64Url(iv)}.${toBase64Url(tag)}.${toBase64Url(ciphertext)}`;
    },
    decodeString(value: string): string {
      if (!value.startsWith(PREFIX)) return value;
      const [ivPart, tagPart, ciphertextPart] = value.slice(PREFIX.length).split(".");
      if (!ivPart || !tagPart || !ciphertextPart) {
        throw new Error("Invalid encrypted memory payload.");
      }
      const decipher = createDecipheriv("aes-256-gcm", key, fromBase64Url(ivPart));
      decipher.setAuthTag(fromBase64Url(tagPart));
      return Buffer.concat([
        decipher.update(fromBase64Url(ciphertextPart)),
        decipher.final(),
      ]).toString("utf8");
    },
  };
}

function toBase64Url(value: Buffer): string {
  return value.toString("base64url");
}

function fromBase64Url(value: string): Buffer {
  return Buffer.from(value, "base64url");
}
