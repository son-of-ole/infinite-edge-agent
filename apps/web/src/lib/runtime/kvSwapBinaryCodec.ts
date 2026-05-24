import type { SerializedKVSwapBlock, SerializedKVSwapPrefillProof } from "./kvSwapPersistence";

export const KV_SWAP_BINARY_MAGIC = 0x564b4549; // IEKV, little-endian.
export const KV_SWAP_BINARY_CODEC_VERSION = 1;
export const MAX_KV_SWAP_BINARY_RECORD_BYTES = 32 * 1024 * 1024;

const HEADER_BYTES = 16;
const STORAGE_RECORD_VERSION = 1;
const MAX_BINARY_STRING_BYTES = 64 * 1024;
const MAX_BINARY_U32_ARRAY_LENGTH = 131_072;
const MAX_BINARY_NUMBER_ARRAY_LENGTH = 131_072;
const MAX_BINARY_MATRIX_ROWS = 65_536;
const MAX_BINARY_MATRIX_CELLS = 4_194_304;
const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

export type KVSwapBinaryDecodeResult =
  | {
      ok: true;
      block: SerializedKVSwapBlock;
      bytesRead: number;
      checksum: string;
    }
  | {
      ok: false;
      reason: string;
    };

export function encodeKVSwapBlockBinary(block: SerializedKVSwapBlock): Uint8Array {
  if (block.version !== STORAGE_RECORD_VERSION) {
    throw new Error(`unsupported_block_version:${String(block.version)}`);
  }

  const payload = new BinaryWriter();
  payload.u32(block.version);
  payload.string(block.namespace);
  payload.string(block.id);
  payload.string(block.modelId);
  payload.string(block.requestId);
  payload.optionalString(block.runtimeBlockId);
  payload.optionalString(block.phase);
  payload.optionalString(block.modelFingerprint);
  payload.optionalString(block.promptTokenHash);
  payload.optionalU32Array(block.promptTokenIds);
  payload.optionalU32(block.prefillTokenCount);
  payload.optionalU32(block.runtimeLayerCount);
  payload.optionalString(block.policyHash);
  payload.u32(block.layer);
  payload.u32(block.startToken);
  payload.u32(block.endToken);
  payload.u8(block.pinned ? 1 : 0);
  payload.f64(block.importance);
  payload.f64(block.estimatedBytes);
  payload.optionalString(block.checksum);
  payload.optionalU32(block.summaryRank);
  payload.compressedSummary(block.compressedKeySummary);
  payload.optionalU32Array(block.tokenIds);
  payload.optionalMatrix(block.queryRows);
  payload.matrix(block.keyRows);
  payload.matrix(block.valueRows);
  payload.optionalMatrix(block.hiddenRows);
  payload.string(block.createdAt);
  payload.string(block.updatedAt);
  payload.f64(block.lastAccessAt);
  payload.f64(block.byteLength);
  payload.optionalString(block.prefillProof ? JSON.stringify(block.prefillProof) : undefined);
  payload.optionalMatrix(block.compactKeyRows);
  payload.optionalMatrix(block.compactValueRows);

  const payloadBytes = payload.toBytes();
  const checksum = checksum32(payloadBytes);
  const output = new Uint8Array(HEADER_BYTES + payloadBytes.byteLength);
  const view = new DataView(output.buffer, output.byteOffset, output.byteLength);
  view.setUint32(0, KV_SWAP_BINARY_MAGIC, true);
  view.setUint16(4, KV_SWAP_BINARY_CODEC_VERSION, true);
  view.setUint16(6, HEADER_BYTES, true);
  view.setUint32(8, payloadBytes.byteLength, true);
  view.setUint32(12, checksum, true);
  output.set(payloadBytes, HEADER_BYTES);
  return output;
}

export function decodeKVSwapBlockBinary(bytes: Uint8Array): KVSwapBinaryDecodeResult {
  if (bytes.byteLength > MAX_KV_SWAP_BINARY_RECORD_BYTES) {
    return { ok: false, reason: `binary_record_too_large:${bytes.byteLength}/${MAX_KV_SWAP_BINARY_RECORD_BYTES}` };
  }
  if (bytes.byteLength < 4) return { ok: false, reason: "truncated_header" };
  const headerView = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const magic = headerView.getUint32(0, true);
  if (magic !== KV_SWAP_BINARY_MAGIC) return { ok: false, reason: "bad_magic" };
  if (bytes.byteLength < HEADER_BYTES) return { ok: false, reason: "truncated_header" };

  const version = headerView.getUint16(4, true);
  if (version !== KV_SWAP_BINARY_CODEC_VERSION) return { ok: false, reason: `unsupported_binary_version:${version}` };

  const headerBytes = headerView.getUint16(6, true);
  if (headerBytes !== HEADER_BYTES) return { ok: false, reason: `unsupported_header_size:${headerBytes}` };

  const payloadLength = headerView.getUint32(8, true);
  if (payloadLength > MAX_KV_SWAP_BINARY_RECORD_BYTES - HEADER_BYTES) {
    return { ok: false, reason: `binary_payload_too_large:${payloadLength}/${MAX_KV_SWAP_BINARY_RECORD_BYTES - HEADER_BYTES}` };
  }
  const expectedLength = HEADER_BYTES + payloadLength;
  if (bytes.byteLength < expectedLength) return { ok: false, reason: `truncated_payload:${bytes.byteLength}/${expectedLength}` };
  if (bytes.byteLength !== expectedLength) return { ok: false, reason: `payload_length_mismatch:${bytes.byteLength}/${expectedLength}` };

  const expectedChecksum = headerView.getUint32(12, true);
  const payload = bytes.subarray(HEADER_BYTES, expectedLength);
  const actualChecksum = checksum32(payload);
  if (actualChecksum !== expectedChecksum) {
    return {
      ok: false,
      reason: `checksum_mismatch:${hex32(expectedChecksum)}:${hex32(actualChecksum)}`,
    };
  }

  try {
    const reader = new BinaryReader(payload);
    const block: SerializedKVSwapBlock = {
      version: reader.u32() as SerializedKVSwapBlock["version"],
      namespace: reader.string(),
      id: reader.string(),
      modelId: reader.string(),
      requestId: reader.string(),
      ...(withOptional("runtimeBlockId", reader.optionalString())),
      ...(withOptional("phase", reader.optionalString() as SerializedKVSwapBlock["phase"] | undefined)),
      ...(withOptional("modelFingerprint", reader.optionalString())),
      ...(withOptional("promptTokenHash", reader.optionalString())),
      ...(withOptional("promptTokenIds", reader.optionalU32Array())),
      ...(withOptional("prefillTokenCount", reader.optionalU32())),
      ...(withOptional("runtimeLayerCount", reader.optionalU32())),
      ...(withOptional("policyHash", reader.optionalString())),
      layer: reader.u32(),
      startToken: reader.u32(),
      endToken: reader.u32(),
      pinned: reader.u8() === 1,
      importance: reader.f64(),
      estimatedBytes: reader.f64(),
      ...(withOptional("checksum", reader.optionalString())),
      ...(withOptional("summaryRank", reader.optionalU32())),
      ...(withOptional("compressedKeySummary", reader.compressedSummary())),
      ...(withOptional("tokenIds", reader.optionalU32Array())),
      ...(withOptional("queryRows", reader.optionalMatrix())),
      keyRows: reader.matrix(),
      valueRows: reader.matrix(),
      ...(withOptional("hiddenRows", reader.optionalMatrix())),
      createdAt: reader.string(),
      updatedAt: reader.string(),
      lastAccessAt: reader.f64(),
      byteLength: reader.f64(),
    };
    if (!reader.done) {
      const prefillProofJson = reader.optionalString();
      if (prefillProofJson !== undefined) {
        const parsed = JSON.parse(prefillProofJson) as SerializedKVSwapPrefillProof;
        block.prefillProof = parsed;
      }
    }
    if (!reader.done) {
      const compactKeyRows = reader.optionalMatrix();
      if (compactKeyRows !== undefined) block.compactKeyRows = compactKeyRows;
    }
    if (!reader.done) {
      const compactValueRows = reader.optionalMatrix();
      if (compactValueRows !== undefined) block.compactValueRows = compactValueRows;
    }
    if (!reader.done) return { ok: false, reason: `trailing_payload_bytes:${reader.remaining}` };
    return {
      ok: true,
      block,
      bytesRead: expectedLength,
      checksum: hex32(actualChecksum),
    };
  } catch (error) {
    if (error instanceof BinaryCodecError) return { ok: false, reason: error.reason };
    return { ok: false, reason: error instanceof Error ? error.message : String(error) };
  }
}

function withOptional<K extends keyof SerializedKVSwapBlock>(
  key: K,
  value: SerializedKVSwapBlock[K] | undefined,
): Pick<SerializedKVSwapBlock, K> | Record<string, never> {
  return value === undefined ? {} : { [key]: value } as Pick<SerializedKVSwapBlock, K>;
}

class BinaryWriter {
  private buffer = new Uint8Array(1024);
  private offset = 0;

  u8(value: number): void {
    this.reserve(1);
    this.buffer[this.offset] = value & 0xff;
    this.offset += 1;
  }

  u16(value: number): void {
    this.reserve(2);
    new DataView(this.buffer.buffer).setUint16(this.offset, checkedUnsigned(value, 0xffff, "u16"), true);
    this.offset += 2;
  }

  u32(value: number): void {
    this.reserve(4);
    new DataView(this.buffer.buffer).setUint32(this.offset, checkedUnsigned(value, 0xffffffff, "u32"), true);
    this.offset += 4;
  }

  f64(value: number): void {
    if (!Number.isFinite(value)) throw new Error("invalid_f64");
    this.reserve(8);
    new DataView(this.buffer.buffer).setFloat64(this.offset, value, true);
    this.offset += 8;
  }

  string(value: string): void {
    const bytes = textEncoder.encode(value);
    this.u32(bytes.byteLength);
    this.bytes(bytes);
  }

  optionalString(value: string | undefined): void {
    this.u8(value === undefined ? 0 : 1);
    if (value !== undefined) this.string(value);
  }

  optionalU32(value: number | undefined): void {
    this.u8(value === undefined ? 0 : 1);
    if (value !== undefined) this.u32(value);
  }

  optionalU32Array(value: number[] | undefined): void {
    this.u8(value === undefined ? 0 : 1);
    if (value !== undefined) this.u32Array(value);
  }

  u32Array(value: number[]): void {
    this.u32(value.length);
    for (const item of value) this.u32(item);
  }

  compressedSummary(value: number[] | string | undefined): void {
    if (value === undefined) {
      this.u8(0);
      return;
    }
    if (typeof value === "string") {
      this.u8(1);
      this.string(value);
      return;
    }
    this.u8(2);
    this.numberArray(value);
  }

  optionalMatrix(value: number[][] | undefined): void {
    this.u8(value === undefined ? 0 : 1);
    if (value !== undefined) this.matrix(value);
  }

  matrix(value: number[][]): void {
    this.u32(value.length);
    for (const row of value) this.numberArray(row);
  }

  numberArray(value: number[]): void {
    this.u32(value.length);
    for (const item of value) this.f64(item);
  }

  toBytes(): Uint8Array {
    return this.buffer.slice(0, this.offset);
  }

  private bytes(bytes: Uint8Array): void {
    this.reserve(bytes.byteLength);
    this.buffer.set(bytes, this.offset);
    this.offset += bytes.byteLength;
  }

  private reserve(byteLength: number): void {
    const required = this.offset + byteLength;
    if (required <= this.buffer.byteLength) return;
    let nextLength = this.buffer.byteLength;
    while (nextLength < required) nextLength *= 2;
    const next = new Uint8Array(nextLength);
    next.set(this.buffer);
    this.buffer = next;
  }
}

class BinaryReader {
  private offset = 0;
  private numericCellsRead = 0;

  constructor(private readonly bytes: Uint8Array) {}

  get done(): boolean {
    return this.offset === this.bytes.byteLength;
  }

  get remaining(): number {
    return this.bytes.byteLength - this.offset;
  }

  u8(): number {
    this.require(1);
    const value = this.bytes[this.offset] ?? 0;
    this.offset += 1;
    return value;
  }

  u32(): number {
    this.require(4);
    const value = new DataView(this.bytes.buffer, this.bytes.byteOffset, this.bytes.byteLength).getUint32(this.offset, true);
    this.offset += 4;
    return value;
  }

  f64(): number {
    this.require(8);
    const value = new DataView(this.bytes.buffer, this.bytes.byteOffset, this.bytes.byteLength).getFloat64(this.offset, true);
    this.offset += 8;
    if (!Number.isFinite(value)) throw new BinaryCodecError("invalid_f64");
    return value;
  }

  string(): string {
    const byteLength = this.u32();
    if (byteLength > MAX_BINARY_STRING_BYTES) {
      throw new BinaryCodecError(`string_too_large:${byteLength}/${MAX_BINARY_STRING_BYTES}`);
    }
    this.require(byteLength);
    const value = textDecoder.decode(this.bytes.subarray(this.offset, this.offset + byteLength));
    this.offset += byteLength;
    return value;
  }

  optionalString(): string | undefined {
    return this.presence() ? this.string() : undefined;
  }

  optionalU32(): number | undefined {
    return this.presence() ? this.u32() : undefined;
  }

  optionalU32Array(): number[] | undefined {
    return this.presence() ? this.u32Array() : undefined;
  }

  u32Array(): number[] {
    const length = this.u32();
    if (length > MAX_BINARY_U32_ARRAY_LENGTH) {
      throw new BinaryCodecError(`u32_array_too_large:${length}/${MAX_BINARY_U32_ARRAY_LENGTH}`);
    }
    const values: number[] = [];
    for (let i = 0; i < length; i += 1) values.push(this.u32());
    return values;
  }

  compressedSummary(): number[] | string | undefined {
    const tag = this.u8();
    if (tag === 0) return undefined;
    if (tag === 1) return this.string();
    if (tag === 2) return this.numberArray();
    throw new BinaryCodecError(`invalid_compressed_summary_tag:${tag}`);
  }

  optionalMatrix(): number[][] | undefined {
    return this.presence() ? this.matrix() : undefined;
  }

  matrix(): number[][] {
    const rows = this.u32();
    if (rows > MAX_BINARY_MATRIX_ROWS) {
      throw new BinaryCodecError(`matrix_rows_too_large:${rows}/${MAX_BINARY_MATRIX_ROWS}`);
    }
    const values: number[][] = [];
    for (let i = 0; i < rows; i += 1) values.push(this.numberArray());
    return values;
  }

  numberArray(): number[] {
    const length = this.u32();
    if (length > MAX_BINARY_NUMBER_ARRAY_LENGTH) {
      throw new BinaryCodecError(`number_array_too_large:${length}/${MAX_BINARY_NUMBER_ARRAY_LENGTH}`);
    }
    this.numericCellsRead += length;
    if (this.numericCellsRead > MAX_BINARY_MATRIX_CELLS) {
      throw new BinaryCodecError(`matrix_cells_too_large:${this.numericCellsRead}/${MAX_BINARY_MATRIX_CELLS}`);
    }
    const values: number[] = [];
    for (let i = 0; i < length; i += 1) values.push(this.f64());
    return values;
  }

  private presence(): boolean {
    const flag = this.u8();
    if (flag === 0) return false;
    if (flag === 1) return true;
    throw new BinaryCodecError(`invalid_presence_flag:${flag}`);
  }

  private require(byteLength: number): void {
    if (this.offset + byteLength > this.bytes.byteLength) {
      throw new BinaryCodecError(`truncated_payload:${this.offset}+${byteLength}/${this.bytes.byteLength}`);
    }
  }
}

class BinaryCodecError extends Error {
  constructor(readonly reason: string) {
    super(reason);
  }
}

function checkedUnsigned(value: number, max: number, label: string): number {
  if (!Number.isInteger(value) || value < 0 || value > max) throw new Error(`invalid_${label}:${String(value)}`);
  return value;
}

function checksum32(bytes: Uint8Array): number {
  let hash = 0x811c9dc5;
  for (const byte of bytes) {
    hash ^= byte;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

function hex32(value: number): string {
  return value.toString(16).padStart(8, "0");
}
