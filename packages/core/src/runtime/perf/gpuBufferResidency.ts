export interface GpuBufferResidencyRecord {
  key: string;
  bytes: number;
  resident: boolean;
  uploads: number;
  cacheHits: number;
}

export class GpuBufferResidencyTracker {
  private readonly records = new Map<string, GpuBufferResidencyRecord>();

  recordUpload(key: string, bytes: number): void {
    const record = this.ensure(key);
    record.bytes = Math.max(record.bytes, Math.max(0, Math.floor(bytes)));
    record.resident = true;
    record.uploads += 1;
  }

  recordCacheHit(key: string): void {
    const record = this.ensure(key);
    record.resident = true;
    record.cacheHits += 1;
  }

  release(key: string): void {
    const record = this.records.get(key);
    if (record) record.resident = false;
  }

  snapshot(): GpuBufferResidencyRecord[] {
    return [...this.records.values()].map((record) => ({ ...record }));
  }

  private ensure(key: string): GpuBufferResidencyRecord {
    const normalized = key.trim() || "unknown";
    let record = this.records.get(normalized);
    if (!record) {
      record = { key: normalized, bytes: 0, resident: false, uploads: 0, cacheHits: 0 };
      this.records.set(normalized, record);
    }
    return record;
  }
}
