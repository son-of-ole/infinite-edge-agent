import * as lancedb from "@lancedb/lancedb";
import type {
  ClusterMetricRecord,
  ConsolidationRunRecord,
  ContextPackTraceRecord,
  GacListOptions,
  GacMemoryStore,
  GacWriteResult,
  IdentityPinRecord,
  MemoryChunk,
  MemoryClusterRecord,
  MemoryContradictionRecord,
  MemoryDeleteOptions,
  MemoryLineageRecord,
  MemoryRepresentativeRecord,
  MemorySearchHit,
  MemorySearchOptions,
  MemorySnapshotStore,
  MemoryStore,
  ModelMemoryActionRecord,
  RawMemoryRecord,
  RawMemorySearchOptions,
  RetrievalAuditRecord,
  RuntimeTrace,
  RuntimeTraceListOptions,
  RuntimeTraceSnapshotStore,
  RuntimeTraceStore,
  SourceDocumentRecord,
  TrainingExampleRecord
} from "@infinite-edge-agent/core";

interface LanceRow {
  id: string;
  tenantId: string;
  cellId: string;
  text: string;
  vector: number[];
  sessionId: string;
  source: string;
  role: string;
  createdAt: string;
  updatedAt: string;
  tagsJson: string;
  metadataJson: string;
  tokenCount: number;
  _distance?: number;
}

interface GacLanceRow {
  id: string;
  gacTable: string;
  tenantId: string;
  cellId: string;
  sessionId: string;
  rawMemoryId: string;
  representativeId: string;
  contextPackId: string;
  createdAt: string;
  recordJson: string;
}

interface RuntimeTraceLanceRow {
  traceId: string;
  requestId: string;
  sessionId: string;
  tenantId: string;
  cellId: string;
  modelId: string;
  backend: string;
  createdAt: string;
  runtimeJson: string;
}

type GacTableName =
  | "raw_memory"
  | "identity_pin"
  | "memory_cluster"
  | "cluster_metric"
  | "memory_representative"
  | "memory_lineage"
  | "consolidation_run"
  | "retrieval_audit"
  | "context_pack_trace"
  | "model_memory_action"
  | "memory_contradiction"
  | "source_document"
  | "training_example";

const MEMORY_REQUIRED_COLUMNS = ["tenantId", "cellId", "role"] as const;
const RUNTIME_TRACE_REQUIRED_COLUMNS = ["tenantId", "cellId"] as const;
const GAC_REQUIRED_COLUMNS = [
  "id",
  "gacTable",
  "tenantId",
  "cellId",
  "sessionId",
  "rawMemoryId",
  "representativeId",
  "contextPackId",
  "createdAt",
  "recordJson"
] as const;
const RUNTIME_TRACE_TABLE = "runtime_traces";
const GAC_TABLE_NAMES: GacTableName[] = [
  "raw_memory",
  "identity_pin",
  "memory_cluster",
  "cluster_metric",
  "memory_representative",
  "memory_lineage",
  "consolidation_run",
  "retrieval_audit",
  "context_pack_trace",
  "model_memory_action",
  "memory_contradiction",
  "source_document",
  "training_example"
];

export interface LanceDatabaseTableStatus {
  name: string;
  exists: boolean;
  rowCount: number | null;
  schema: Array<{ name: string; type: string }>;
}

export interface LanceMemoryTableStatus extends LanceDatabaseTableStatus {
  vectorDimension: number | null;
  requiredColumns: Record<(typeof MEMORY_REQUIRED_COLUMNS)[number], boolean>;
}

export interface LanceDatabaseStatus {
  ok: true;
  repaired: boolean;
  repairs: string[];
  memoryTable: LanceMemoryTableStatus;
  runtimeTraceTable: LanceDatabaseTableStatus;
  gacTables: LanceDatabaseTableStatus[];
  backupTables: LanceDatabaseTableStatus[];
}

export interface LanceDatabaseStatusOptions {
  repair?: boolean;
  expectedVectorDimension?: number;
}

export interface LanceStringCodec {
  encodeString(value: string): string;
  decodeString(value: string): string;
}

export interface LanceMemoryStoreOptions {
  stringCodec?: LanceStringCodec;
}

export class LanceMemoryStore implements MemoryStore, MemorySnapshotStore, RuntimeTraceStore, RuntimeTraceSnapshotStore, GacMemoryStore {
  private dbPromise: Promise<any>;
  private tablePromise: Promise<any> | null = null;
  private runtimeTraceTablePromise: Promise<any> | null = null;
  private gacTablePromises = new Map<string, Promise<any>>();
  private readonly stringCodec: LanceStringCodec | undefined;

  constructor(
    private readonly uri: string,
    private readonly tableName: string,
    options: LanceMemoryStoreOptions = {}
  ) {
    this.dbPromise = lancedb.connect(uri);
    this.stringCodec = options.stringCodec;
  }

  async upsert(chunks: MemoryChunk[]): Promise<void> {
    if (chunks.length === 0) return;
    const rows = dedupeMemoryRows(chunks.map((chunk) => toLanceRow(chunk, this.stringCodec)));
    const { table, created } = await this.ensureTable(rows);
    if (!created) {
      await this.deleteExistingMemoryRows(table, rows);
      await table.add(rows);
    }
  }

  async search(queryEmbedding: number[], options: MemorySearchOptions = {}): Promise<MemorySearchHit[]> {
    const table = await this.openTableIfExists();
    if (!table) return [];
    const limit = options.limit ?? 8;
    let query = table.vectorSearch(queryEmbedding).limit(limit * 3);

    const filters = buildMemoryFilters(options);
    if (filters.length > 0) query = query.where(filters.join(" AND "));

    const rows = (await query.toArray()) as LanceRow[];
    const now = Date.now();
    const minScore = options.minScore ?? -1;
    const tags = options.tags ?? [];

    return rows
      .map((row) => fromLanceRow(row, this.stringCodec))
      .filter((hit) => tags.length === 0 || tags.every((tag) => hit.tags.includes(tag)))
      .filter((hit) => {
        if (!options.maxAgeMs) return true;
        return now - new Date(hit.createdAt).getTime() <= options.maxAgeMs;
      })
      .filter((hit) => hit.score >= minScore)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  }

  async clear(): Promise<void> {
    const db = await this.dbPromise;
    try {
      await db.dropTable(this.tableName);
    } catch {
      // Table may not exist yet.
    }
    this.tablePromise = null;
  }

  async deleteMemory(options: MemoryDeleteOptions): Promise<number> {
    assertTargetedDelete(options);
    const table = await this.openTableIfExists();
    if (!table) {
      await this.deleteGacRecordsForMemoryDelete(options, []);
      return 0;
    }

    const filters = buildMemoryFilters(options);
    if (!options.tags || options.tags.length === 0) {
      const predicate = filters.join(" AND ");
      const rowsToDelete = await listRowsForPredicate<LanceRow>(table, predicate);
      const count = await table.countRows(predicate);
      if (count > 0) await table.delete(predicate);
      await this.deleteGacRecordsForMemoryDelete(options, rowsToDelete);
      return count;
    }

    let query = table.query();
    if (filters.length > 0) query = query.where(filters.join(" AND "));
    const rows = (await query.toArray()) as LanceRow[];
    const rowsToDelete = rows.filter((row) => {
      const chunk = fromLanceRow(row, this.stringCodec);
      return options.tags?.every((tag) => chunk.tags.includes(tag));
    });
    const ids = rowsToDelete.map((row) => row.id);
    if (ids.length === 0) return 0;

    const idPredicate = `id IN (${ids.map((id) => `'${escapeSqlString(id)}'`).join(", ")})`;
    await table.delete([...filters, idPredicate].join(" AND "));
    await this.deleteGacRecordsForMemoryDelete(options, rowsToDelete);
    return ids.length;
  }

  async listMemoryChunks(options: { sessionId?: string; limit?: number; tenantId?: string; cellId?: string } = {}): Promise<MemoryChunk[]> {
    const table = await this.openTableIfExists();
    if (!table) return [];
    let query = table.query();
    const filters = buildMemoryFilters(options);
    if (filters.length > 0) query = query.where(filters.join(" AND "));
    if (options.limit !== undefined) query = query.limit(options.limit);
    const rows = (await query.toArray()) as LanceRow[];
    return rows.map((row) => {
      const { score: _score, ...chunk } = fromLanceRow(row, this.stringCodec);
      return chunk;
    });
  }

  async importMemoryChunks(chunks: MemoryChunk[]): Promise<void> {
    await this.upsert(chunks);
  }

  async writeRawMemory(records: RawMemoryRecord[]): Promise<GacWriteResult> {
    return this.writeGacRecords("raw_memory", records, toRawMemoryRow);
  }

  async listRawMemory(options: GacListOptions = {}): Promise<RawMemoryRecord[]> {
    return this.listGacRecords<RawMemoryRecord>("raw_memory", options);
  }

  async searchRawMemory(options: RawMemorySearchOptions = {}): Promise<RawMemoryRecord[]> {
    const limit = options.limit ?? 8;
    const queryText = options.queryText?.trim().toLocaleLowerCase();
    const { limit: _limit, queryText: _queryText, includeDeleted: _includeDeleted, ...listOptions } = options;
    const records = await this.listGacRecords<RawMemoryRecord>("raw_memory", listOptions);
    return records
      .filter((record) => options.includeDeleted || !record.deletedAt && record.retentionClass !== "user_deleted")
      .filter((record) => {
        if (!queryText) return true;
        return [
          record.text,
          record.canonicalText,
          record.memoryKind,
          record.sourceType,
          record.hash
        ].filter(Boolean).some((value) => String(value).toLocaleLowerCase().includes(queryText));
      })
      .sort((a, b) => {
        const importanceDelta = b.importance - a.importance;
        if (importanceDelta !== 0) return importanceDelta;
        return b.updatedAt.localeCompare(a.updatedAt);
      })
      .slice(0, limit);
  }

  async writeIdentityPins(records: IdentityPinRecord[]): Promise<GacWriteResult> {
    return this.writeGacRecords("identity_pin", records, toIdentityPinRow);
  }

  async listIdentityPins(options: GacListOptions = {}): Promise<IdentityPinRecord[]> {
    return this.listGacRecords<IdentityPinRecord>("identity_pin", options);
  }

  async writeMemoryClusters(records: MemoryClusterRecord[]): Promise<GacWriteResult> {
    return this.writeGacRecords("memory_cluster", records, toMemoryClusterRow);
  }

  async listMemoryClusters(options: GacListOptions = {}): Promise<MemoryClusterRecord[]> {
    return this.listGacRecords<MemoryClusterRecord>("memory_cluster", options);
  }

  async writeClusterMetrics(records: ClusterMetricRecord[]): Promise<GacWriteResult> {
    return this.writeGacRecords("cluster_metric", records, toClusterMetricRow);
  }

  async listClusterMetrics(options: GacListOptions = {}): Promise<ClusterMetricRecord[]> {
    return this.listGacRecords<ClusterMetricRecord>("cluster_metric", options);
  }

  async writeMemoryRepresentatives(
    records: MemoryRepresentativeRecord[],
    lineage: MemoryLineageRecord[] = [],
  ): Promise<GacWriteResult> {
    assertRepresentativeLineage(records, lineage);
    if (lineage.length > 0) await this.writeMemoryLineage(lineage);
    return this.writeGacRecords("memory_representative", records, toRepresentativeRow);
  }

  async listMemoryRepresentatives(options: GacListOptions = {}): Promise<MemoryRepresentativeRecord[]> {
    return this.listGacRecords<MemoryRepresentativeRecord>("memory_representative", options);
  }

  async writeMemoryLineage(records: MemoryLineageRecord[]): Promise<GacWriteResult> {
    return this.writeGacRecords("memory_lineage", records, toLineageRow);
  }

  async listMemoryLineage(options: GacListOptions = {}): Promise<MemoryLineageRecord[]> {
    return this.listGacRecords<MemoryLineageRecord>("memory_lineage", options);
  }

  async writeConsolidationRuns(records: ConsolidationRunRecord[]): Promise<GacWriteResult> {
    return this.writeGacRecords("consolidation_run", records, toConsolidationRunRow);
  }

  async listConsolidationRuns(options: GacListOptions = {}): Promise<ConsolidationRunRecord[]> {
    return this.listGacRecords<ConsolidationRunRecord>("consolidation_run", options);
  }

  async writeRetrievalAudits(records: RetrievalAuditRecord[]): Promise<GacWriteResult> {
    return this.writeGacRecords("retrieval_audit", records, toRetrievalAuditRow);
  }

  async listRetrievalAudits(options: GacListOptions = {}): Promise<RetrievalAuditRecord[]> {
    return this.listGacRecords<RetrievalAuditRecord>("retrieval_audit", options);
  }

  async writeContextPackTraces(records: ContextPackTraceRecord[]): Promise<GacWriteResult> {
    return this.writeGacRecords("context_pack_trace", records, toContextPackTraceRow);
  }

  async listContextPackTraces(options: GacListOptions = {}): Promise<ContextPackTraceRecord[]> {
    return this.listGacRecords<ContextPackTraceRecord>("context_pack_trace", options);
  }

  async writeModelMemoryActions(records: ModelMemoryActionRecord[]): Promise<GacWriteResult> {
    return this.writeGacRecords("model_memory_action", records, toModelMemoryActionRow);
  }

  async listModelMemoryActions(options: GacListOptions = {}): Promise<ModelMemoryActionRecord[]> {
    return this.listGacRecords<ModelMemoryActionRecord>("model_memory_action", options);
  }

  async writeMemoryContradictions(records: MemoryContradictionRecord[]): Promise<GacWriteResult> {
    return this.writeGacRecords("memory_contradiction", records, toMemoryContradictionRow);
  }

  async listMemoryContradictions(options: GacListOptions = {}): Promise<MemoryContradictionRecord[]> {
    return this.listGacRecords<MemoryContradictionRecord>("memory_contradiction", options);
  }

  async writeSourceDocuments(records: SourceDocumentRecord[]): Promise<GacWriteResult> {
    return this.writeGacRecords("source_document", records, toSourceDocumentRow);
  }

  async listSourceDocuments(options: GacListOptions = {}): Promise<SourceDocumentRecord[]> {
    return this.listGacRecords<SourceDocumentRecord>("source_document", options);
  }

  async writeTrainingExamples(records: TrainingExampleRecord[]): Promise<GacWriteResult> {
    return this.writeGacRecords("training_example", records, toTrainingExampleRow);
  }

  async listTrainingExamples(options: GacListOptions = {}): Promise<TrainingExampleRecord[]> {
    return this.listGacRecords<TrainingExampleRecord>("training_example", options);
  }

  async writeRuntimeTrace(trace: RuntimeTrace): Promise<void> {
    const rows = [toRuntimeTraceRow(trace, this.stringCodec)];
    const row = rows[0]!;
    const { table, created } = await this.ensureRuntimeTraceTable(rows);
    if (!created) {
      await table.delete(buildRuntimeTraceIdPredicate(row));
      await table.add(rows);
    }
  }

  async listRuntimeTraces(options: RuntimeTraceListOptions = {}): Promise<RuntimeTrace[]> {
    const table = await this.openRuntimeTraceTableIfExists();
    if (!table) return [];
    let query = table.query();
    const filters = buildRuntimeTraceFilters(options);
    if (filters.length > 0) query = query.where(filters.join(" AND "));
    const rows = (await query.toArray()) as RuntimeTraceLanceRow[];
    return rows
      .map((row) => fromRuntimeTraceRow(row, this.stringCodec))
      .sort((a, b) => b.createdAt.localeCompare(a.createdAt))
      .slice(0, options.limit ?? 20);
  }

  async importRuntimeTraces(traces: RuntimeTrace[]): Promise<void> {
    for (const trace of traces) {
      await this.writeRuntimeTrace(trace);
    }
  }

  async getDatabaseStatus(options: LanceDatabaseStatusOptions = {}): Promise<LanceDatabaseStatus> {
    const db = await this.dbPromise;
    const tableNames = await this.getTableNames(db);
    const repairs = options.repair ? await this.repairDatabase(db, tableNames, options.expectedVectorDimension) : [];
    const refreshedTableNames = repairs.length > 0 ? await this.getTableNames(db) : tableNames;
    return {
      ok: true,
      repaired: repairs.length > 0,
      repairs,
      memoryTable: await this.inspectMemoryTable(db, this.tableName, refreshedTableNames),
      runtimeTraceTable: await this.inspectTable(db, RUNTIME_TRACE_TABLE, refreshedTableNames),
      gacTables: await Promise.all(GAC_TABLE_NAMES.map((name) => this.inspectTable(db, name, refreshedTableNames))),
      backupTables: await Promise.all(
        refreshedTableNames
          .filter((name) => name.startsWith(`${this.tableName}_incompatible_`))
          .sort()
          .map((name) => this.inspectTable(db, name, refreshedTableNames))
      )
    };
  }

  private async ensureTable(initialRows: LanceRow[]): Promise<{ table: any; created: boolean }> {
    const existing = await this.openTableIfExists();
    if (existing) {
      const expectedVectorDimension = initialRows[0]?.vector.length;
      if (expectedVectorDimension !== undefined && await this.tableVectorDimensionMismatch(existing, expectedVectorDimension)) {
        await this.recreateMemoryTable();
      } else {
        return { table: existing, created: false };
      }
    }
    const db = await this.dbPromise;
    this.tablePromise = db.createTable(this.tableName, initialRows);
    return { table: await this.tablePromise, created: true };
  }

  private async deleteExistingMemoryRows(table: any, rows: LanceRow[]): Promise<void> {
    for (const row of rows) {
      await table.delete(buildScopedMemoryIdPredicate(row));
    }
  }

  private async openTableIfExists(): Promise<any | null> {
    if (this.tablePromise) return this.tablePromise;
    const db = await this.dbPromise;
    try {
      const table = await db.openTable(this.tableName);
      await this.ensureMemoryTableScopeColumns(table);
      this.tablePromise = Promise.resolve(table);
      return table;
    } catch {
      return null;
    }
  }

  private async ensureMemoryTableScopeColumns(table: any): Promise<void> {
    if (typeof table.schema !== "function" || typeof table.addColumns !== "function") return;
    const schema = await table.schema();
    const fieldNames = new Set((schema.fields ?? []).map((field: { name: string }) => field.name));
    const missing = MEMORY_REQUIRED_COLUMNS.filter((column) => !fieldNames.has(column));
    if (missing.length === 0) return;
    await table.addColumns(missing.map((name) => ({ name, valueSql: "''" })));
  }

  private async tableVectorDimensionMismatch(table: any, expectedDimension: number): Promise<boolean> {
    if (typeof table.schema !== "function") return false;
    const schema = await table.schema();
    const vectorField = (schema.fields ?? []).find((field: { name: string }) => field.name === "vector");
    const actualDimension = parseFixedVectorDimension(String(vectorField?.type ?? ""));
    return actualDimension !== undefined && actualDimension !== expectedDimension;
  }

  private async recreateMemoryTable(tableOverride?: any): Promise<void> {
    const db = await this.dbPromise;
    const table = tableOverride ?? (this.tablePromise ? await this.tablePromise : null);
    if (table) {
      const rows = ((await table.query().toArray()) as LanceRow[]).map(toPlainLanceRow);
      if (rows.length > 0) {
        const backupName = await this.nextBackupTableName(db);
        await db.createTable(backupName, rows);
      }
    }
    await db.dropTable(this.tableName);
    this.tablePromise = null;
  }

  private async nextBackupTableName(db: any): Promise<string> {
    const tableNames = typeof db.tableNames === "function" ? await db.tableNames() : [];
    const base = `${this.tableName}_incompatible_${new Date().toISOString().replace(/[^0-9]/g, "").slice(0, 14)}`;
    if (!tableNames.includes(base)) return base;
    let suffix = 2;
    while (tableNames.includes(`${base}_${suffix}`)) suffix += 1;
    return `${base}_${suffix}`;
  }

  private async writeGacRecords<T>(tableName: GacTableName, records: T[], toRow: (record: T) => GacLanceRow): Promise<GacWriteResult> {
    const traceId = makeWriteTraceId(tableName);
    if (records.length === 0) return { ok: true, count: 0, traceId };
    const rows = records.map((record) => {
      const row = { ...toRow(record), gacTable: tableName };
      return { ...row, recordJson: this.encodeString(row.recordJson) };
    });
    const { table, created } = await this.ensureGacTable(tableName, rows);
    if (!created) {
      await this.deleteExistingGacRows(table, rows);
      await table.add(rows);
    }
    return { ok: true, count: records.length, traceId };
  }

  private async listGacRecords<T>(tableName: GacTableName, options: GacListOptions): Promise<T[]> {
    const table = await this.openGacTableIfExists(tableName);
    if (!table) return [];
    let query = table.query();
    const filters = [`gacTable = '${escapeSqlString(tableName)}'`, ...buildGacFilters(options)];
    if (filters.length > 0) query = query.where(filters.join(" AND "));
    if (options.limit !== undefined) query = query.limit(options.limit);
    const rows = (await query.toArray()) as GacLanceRow[];
    return rows.map((row) => safeParseJson<T>(this.decodeString(row.recordJson), null as T)).filter((record) => record !== null);
  }

  private async ensureGacTable(tableName: GacTableName, initialRows: GacLanceRow[]): Promise<{ table: any; created: boolean }> {
    const existing = await this.openGacTableIfExists(tableName);
    if (existing) {
      await this.ensureGacTableColumns(tableName, existing);
      return { table: existing, created: false };
    }
    const db = await this.dbPromise;
    const promise = db.createTable(tableName, initialRows);
    this.gacTablePromises.set(tableName, promise);
    return { table: await promise, created: true };
  }

  private async openGacTableIfExists(tableName: GacTableName): Promise<any | null> {
    const existing = this.gacTablePromises.get(tableName);
    if (existing) {
      const table = await existing;
      await this.ensureGacTableColumns(tableName, table);
      return table;
    }
    const db = await this.dbPromise;
    try {
      const promise = db.openTable(tableName);
      const table = await promise;
      this.gacTablePromises.set(tableName, Promise.resolve(table));
      await this.ensureGacTableColumns(tableName, table);
      return table;
    } catch {
      this.gacTablePromises.delete(tableName);
      return null;
    }
  }

  private async deleteExistingGacRows(table: any, rows: GacLanceRow[]): Promise<void> {
    for (const row of rows) {
      await table.delete(buildScopedGacIdPredicate(row));
    }
  }

  private async ensureRuntimeTraceTable(initialRows: RuntimeTraceLanceRow[]): Promise<{ table: any; created: boolean }> {
    const existing = await this.openRuntimeTraceTableIfExists();
    if (existing) {
      await this.ensureRuntimeTraceTableScopeColumns(existing);
      return { table: existing, created: false };
    }
    const db = await this.dbPromise;
    this.runtimeTraceTablePromise = db.createTable(RUNTIME_TRACE_TABLE, initialRows);
    return { table: await this.runtimeTraceTablePromise, created: true };
  }

  private async openRuntimeTraceTableIfExists(): Promise<any | null> {
    if (this.runtimeTraceTablePromise) return this.runtimeTraceTablePromise;
    const db = await this.dbPromise;
    try {
      const table = await db.openTable(RUNTIME_TRACE_TABLE);
      this.runtimeTraceTablePromise = Promise.resolve(table);
      await this.ensureRuntimeTraceTableScopeColumns(table);
      return table;
    } catch {
      return null;
    }
  }

  private async ensureRuntimeTraceTableScopeColumns(table: any): Promise<void> {
    if (typeof table.schema !== "function" || typeof table.addColumns !== "function") return;
    const schema = await table.schema();
    const fieldNames = new Set((schema.fields ?? []).map((field: { name: string }) => field.name));
    const missing = RUNTIME_TRACE_REQUIRED_COLUMNS.filter((column) => !fieldNames.has(column));
    if (missing.length === 0) return;
    await table.addColumns(missing.map((name) => ({ name, valueSql: "''" })));
  }

  private async ensureGacTableColumns(tableName: GacTableName, table: any): Promise<boolean> {
    if (typeof table.schema !== "function" || typeof table.addColumns !== "function") return false;
    const schema = await table.schema();
    const fieldNames = new Set((schema.fields ?? []).map((field: { name: string }) => field.name));
    const missing = GAC_REQUIRED_COLUMNS.filter((column) => !fieldNames.has(column));
    if (missing.length === 0) return false;
    await table.addColumns(missing.map((name) => ({
      name,
      valueSql: name === "gacTable" ? `'${escapeSqlString(tableName)}'` : "''"
    })));
    return true;
  }

  private async repairDatabase(db: any, tableNames: string[], expectedVectorDimension?: number): Promise<string[]> {
    const repairs: string[] = [];
    if (tableNames.includes(this.tableName)) {
      repairs.push(...await this.repairMemoryTable(db, expectedVectorDimension));
    }
    if (tableNames.includes(RUNTIME_TRACE_TABLE) && await this.repairRuntimeTraceTable(db)) {
      repairs.push(`${RUNTIME_TRACE_TABLE}:columns`);
    }
    for (const tableName of GAC_TABLE_NAMES) {
      if (tableNames.includes(tableName) && await this.repairGacTable(db, tableName)) {
        repairs.push(`${tableName}:columns`);
      }
    }
    return repairs;
  }

  private async repairMemoryTable(db: any, expectedVectorDimension?: number): Promise<string[]> {
    const table = await db.openTable(this.tableName);
    if (expectedVectorDimension !== undefined && await this.tableVectorDimensionMismatch(table, expectedVectorDimension)) {
      await this.recreateMemoryTable(table);
      return [`${this.tableName}:vector-dimension`];
    }
    const schemaBefore = await readTableSchema(table);
    const fieldsBefore = new Set(schemaBefore.map((field) => field.name));
    const missingBefore = MEMORY_REQUIRED_COLUMNS.filter((column) => !fieldsBefore.has(column));
    await this.ensureMemoryTableScopeColumns(table);
    if (missingBefore.length > 0) this.tablePromise = Promise.resolve(table);
    return missingBefore.length > 0 ? [`${this.tableName}:columns`] : [];
  }

  private async repairRuntimeTraceTable(db: any): Promise<boolean> {
    const table = await db.openTable(RUNTIME_TRACE_TABLE);
    const schemaBefore = await readTableSchema(table);
    const fieldsBefore = new Set(schemaBefore.map((field) => field.name));
    const missingBefore = RUNTIME_TRACE_REQUIRED_COLUMNS.filter((column) => !fieldsBefore.has(column));
    await this.ensureRuntimeTraceTableScopeColumns(table);
    if (missingBefore.length > 0) this.runtimeTraceTablePromise = Promise.resolve(table);
    return missingBefore.length > 0;
  }

  private async repairGacTable(db: any, tableName: GacTableName): Promise<boolean> {
    const table = await db.openTable(tableName);
    const repaired = await this.ensureGacTableColumns(tableName, table);
    if (repaired) this.gacTablePromises.set(tableName, Promise.resolve(table));
    return repaired;
  }

  private async deleteGacRecordsForMemoryDelete(options: MemoryDeleteOptions, rowsToDelete: LanceRow[]): Promise<number> {
    const rawMemoryIds = new Set(rowsToDelete.map((row) => row.id));
    const baseOptions: GacListOptions = {
      ...(options.tenantId ? { tenantId: options.tenantId } : {}),
      ...(options.cellId ? { cellId: options.cellId } : {}),
      ...(options.sessionId ? { sessionId: options.sessionId } : {})
    };
    const deleteEntireScope = Boolean(baseOptions.tenantId || baseOptions.cellId || baseOptions.sessionId);
    let deletedCount = 0;
    const rowsByTable = new Map<GacTableName, GacLanceRow[]>();
    const representativeIds = new Set<string>();

    for (const tableName of GAC_TABLE_NAMES) {
      const table = await this.openGacTableIfExists(tableName);
      if (!table) continue;
      const rows = await this.queryGacRowsForDelete(table, baseOptions);
      rowsByTable.set(tableName, rows);
      if (deleteEntireScope) {
        rows
          .filter((row) => row.representativeId)
          .forEach((row) => representativeIds.add(row.representativeId));
        continue;
      }
      for (const row of rows) {
        if (gacRowReferencesRawMemory(row, rawMemoryIds, (value) => this.decodeString(value))) {
          if (row.representativeId) representativeIds.add(row.representativeId);
          const parsed = safeParseJson<Record<string, unknown>>(this.decodeString(row.recordJson), {});
          const recordRepresentativeId = getStringField(parsed, "id") ?? getStringField(parsed, "representativeId");
          if (recordRepresentativeId && tableName === "memory_representative") representativeIds.add(recordRepresentativeId);
        }
      }
    }

    for (const tableName of GAC_TABLE_NAMES) {
      const table = await this.openGacTableIfExists(tableName);
      if (!table) continue;
      const rows = rowsByTable.get(tableName) ?? [];
      const ids = rows
        .filter((row) =>
          deleteEntireScope
          || gacRowReferencesRawMemory(row, rawMemoryIds, (value) => this.decodeString(value))
          || gacRowReferencesRepresentative(row, representativeIds, (value) => this.decodeString(value))
        )
        .map((row) => row.id);
      if (ids.length === 0) continue;
      await table.delete(buildGacIdListPredicate(ids, baseOptions));
      deletedCount += ids.length;
    }

    return deletedCount;
  }

  private async queryGacRowsForDelete(table: any, options: GacListOptions): Promise<GacLanceRow[]> {
    let query = table.query();
    const filters = buildGacFilters(options);
    if (filters.length > 0) query = query.where(filters.join(" AND "));
    return (await query.toArray()) as GacLanceRow[];
  }

  private async inspectMemoryTable(db: any, name: string, tableNames: string[]): Promise<LanceMemoryTableStatus> {
    const table = await this.openNamedTable(db, name, tableNames);
    if (!table) return {
      name,
      exists: false,
      rowCount: null,
      schema: [],
      vectorDimension: null,
      requiredColumns: { tenantId: false, cellId: false, role: false }
    };
    const schema = await readTableSchema(table);
    const fieldNames = new Set(schema.map((field) => field.name));
    const vectorField = schema.find((field) => field.name === "vector");
    return {
      name,
      exists: true,
      rowCount: await countTableRows(table),
      schema,
      vectorDimension: vectorField ? parseFixedVectorDimension(vectorField.type) ?? null : null,
      requiredColumns: {
        tenantId: fieldNames.has("tenantId"),
        cellId: fieldNames.has("cellId"),
        role: fieldNames.has("role")
      }
    };
  }

  private async inspectTable(db: any, name: string, tableNames: string[]): Promise<LanceDatabaseTableStatus> {
    const table = await this.openNamedTable(db, name, tableNames);
    if (!table) return { name, exists: false, rowCount: null, schema: [] };
    return {
      name,
      exists: true,
      rowCount: await countTableRows(table),
      schema: await readTableSchema(table)
    };
  }

  private async openNamedTable(db: any, name: string, tableNames: string[]): Promise<any | null> {
    if (!tableNames.includes(name)) return null;
    try {
      return await db.openTable(name);
    } catch {
      return null;
    }
  }

  private async getTableNames(db: any): Promise<string[]> {
    return typeof db.tableNames === "function" ? await db.tableNames() : [];
  }

  private encodeString(value: string): string {
    return this.stringCodec ? this.stringCodec.encodeString(value) : value;
  }

  private decodeString(value: string): string {
    return this.stringCodec ? this.stringCodec.decodeString(value) : value;
  }
}

export class InMemoryRuntimeTraceStore implements RuntimeTraceStore {
  private readonly traces = new Map<string, RuntimeTrace>();

  async writeRuntimeTrace(trace: RuntimeTrace): Promise<void> {
    this.traces.set(trace.traceId, trace);
  }

  async listRuntimeTraces(options: RuntimeTraceListOptions = {}): Promise<RuntimeTrace[]> {
    return [...this.traces.values()]
      .filter((trace) => !options.sessionId || trace.sessionId === options.sessionId)
      .filter((trace) => !options.tenantId || trace.tenantId === options.tenantId)
      .filter((trace) => !options.cellId || trace.cellId === options.cellId)
      .sort((a, b) => b.createdAt.localeCompare(a.createdAt))
      .slice(0, options.limit ?? 20);
  }
}

function toLanceRow(chunk: MemoryChunk, codec?: LanceStringCodec): LanceRow {
  return {
    id: chunk.id,
    tenantId: getStringMetadata(chunk.metadata, "edgeTenantId"),
    cellId: getStringMetadata(chunk.metadata, "edgeCellId"),
    text: encodeString(chunk.text, codec),
    vector: chunk.embedding,
    sessionId: chunk.sessionId,
    source: chunk.source,
    role: chunk.role ?? "",
    createdAt: chunk.createdAt,
    updatedAt: chunk.updatedAt,
    tagsJson: JSON.stringify(chunk.tags),
    metadataJson: encodeString(JSON.stringify(chunk.metadata), codec),
    tokenCount: chunk.tokenCount
  };
}

function dedupeMemoryRows(rows: LanceRow[]): LanceRow[] {
  return [...new Map(rows.map((row) => [scopedMemoryRowKey(row), row])).values()];
}

function scopedMemoryRowKey(row: LanceRow): string {
  return `${row.tenantId}\0${row.cellId}\0${row.id}`;
}

function buildScopedMemoryIdPredicate(row: LanceRow): string {
  return [
    `id = '${escapeSqlString(row.id)}'`,
    `tenantId = '${escapeSqlString(row.tenantId)}'`,
    `cellId = '${escapeSqlString(row.cellId)}'`
  ].join(" AND ");
}

function buildScopedGacIdPredicate(row: GacLanceRow): string {
  return [
    `id = '${escapeSqlString(row.id)}'`,
    `tenantId = '${escapeSqlString(row.tenantId)}'`,
    `cellId = '${escapeSqlString(row.cellId)}'`
  ].join(" AND ");
}

function buildGacIdListPredicate(ids: string[], options: Pick<GacListOptions, "tenantId" | "cellId" | "sessionId">): string {
  return [
    `id IN (${ids.map((id) => `'${escapeSqlString(id)}'`).join(", ")})`,
    ...buildGacFilters(options)
  ].join(" AND ");
}

function toPlainLanceRow(row: LanceRow): LanceRow {
  return {
    id: row.id,
    tenantId: row.tenantId ?? "",
    cellId: row.cellId ?? "",
    text: row.text,
    vector: Array.from(row.vector),
    sessionId: row.sessionId,
    source: row.source,
    role: row.role ?? "",
    createdAt: row.createdAt,
    updatedAt: row.updatedAt,
    tagsJson: row.tagsJson,
    metadataJson: row.metadataJson,
    tokenCount: row.tokenCount
  };
}

function fromLanceRow(row: LanceRow, codec?: LanceStringCodec): MemorySearchHit {
  const distance = row._distance ?? 0;
  return {
    id: row.id,
    text: decodeString(row.text ?? "", codec),
    embedding: row.vector,
    sessionId: row.sessionId,
    source: row.source as MemoryChunk["source"],
    ...(row.role ? { role: row.role as NonNullable<MemoryChunk["role"]> } : {}),
    createdAt: row.createdAt,
    updatedAt: row.updatedAt,
    tags: safeParseJson<string[]>(row.tagsJson, []),
    metadata: {
      ...safeParseJson<Record<string, unknown>>(decodeString(row.metadataJson ?? "{}", codec), {}),
      ...(row.tenantId ? { edgeTenantId: row.tenantId } : {}),
      ...(row.cellId ? { edgeCellId: row.cellId } : {})
    },
    tokenCount: row.tokenCount,
    score: 1 / (1 + Math.max(0, distance))
  };
}

function toRuntimeTraceRow(trace: RuntimeTrace, codec?: LanceStringCodec): RuntimeTraceLanceRow {
  return {
    traceId: trace.traceId,
    requestId: trace.requestId,
    sessionId: trace.sessionId,
    tenantId: trace.tenantId ?? "",
    cellId: trace.cellId ?? "",
    modelId: trace.modelId,
    backend: trace.backend,
    createdAt: trace.createdAt,
    runtimeJson: encodeString(JSON.stringify(trace.runtime), codec)
  };
}

function fromRuntimeTraceRow(row: RuntimeTraceLanceRow, codec?: LanceStringCodec): RuntimeTrace {
  return {
    traceId: row.traceId,
    requestId: row.requestId,
    sessionId: row.sessionId,
    ...(row.tenantId ? { tenantId: row.tenantId } : {}),
    ...(row.cellId ? { cellId: row.cellId } : {}),
    modelId: row.modelId,
    backend: row.backend,
    createdAt: row.createdAt,
    runtime: safeParseJson<Record<string, unknown>>(decodeString(row.runtimeJson, codec), {})
  };
}

async function readTableSchema(table: any): Promise<Array<{ name: string; type: string }>> {
  if (typeof table.schema !== "function") return [];
  const schema = await table.schema();
  return (schema.fields ?? []).map((field: { name: string; type?: unknown }) => ({
    name: field.name,
    type: String(field.type ?? "")
  }));
}

async function countTableRows(table: any): Promise<number | null> {
  if (typeof table.countRows === "function") {
    try {
      return await table.countRows();
    } catch {
      // Some LanceDB versions only support countRows with a predicate.
    }
  }
  try {
    const rows = await table.query().toArray();
    return rows.length;
  } catch {
    return null;
  }
}

async function listRowsForPredicate<T>(table: any, predicate: string): Promise<T[]> {
  let query = table.query();
  if (predicate) query = query.where(predicate);
  return (await query.toArray()) as T[];
}

function safeParseJson<T>(value: string, fallback: T): T {
  try {
    return JSON.parse(value) as T;
  } catch {
    return fallback;
  }
}

function encodeString(value: string, codec?: LanceStringCodec): string {
  return codec ? codec.encodeString(value) : value;
}

function decodeString(value: string, codec?: LanceStringCodec): string {
  return codec ? codec.decodeString(value) : value;
}

function assertTargetedDelete(options: MemoryDeleteOptions): void {
  if (!options.sessionId && !options.tenantId && !options.cellId && (!options.tags || options.tags.length === 0)) {
    throw new Error("deleteMemory requires a sessionId, tenant/cell scope, or at least one tag.");
  }
}

function buildMemoryFilters(options: Pick<MemoryDeleteOptions, "sessionId" | "tenantId" | "cellId">): string[] {
  const filters: string[] = [];
  if (options.sessionId) filters.push(`sessionId = '${escapeSqlString(options.sessionId)}'`);
  if (options.tenantId) filters.push(`tenantId = '${escapeSqlString(options.tenantId)}'`);
  if (options.cellId) filters.push(`cellId = '${escapeSqlString(options.cellId)}'`);
  return filters;
}

function getStringMetadata(metadata: Record<string, unknown>, key: string): string {
  const value = metadata[key];
  return typeof value === "string" ? value : "";
}

function toRawMemoryRow(record: RawMemoryRecord): GacLanceRow {
  return toGacRow(record.id, record, {
    tenantId: record.tenantId,
    cellId: record.cellId,
    ...(record.sessionId ? { sessionId: record.sessionId } : {}),
    rawMemoryId: record.id,
    createdAt: record.createdAt
  });
}

function toIdentityPinRow(record: IdentityPinRecord): GacLanceRow {
  return toGacRow(record.id, record, {
    tenantId: record.tenantId,
    cellId: record.cellId,
    ...(record.sessionId ? { sessionId: record.sessionId } : {}),
    rawMemoryId: record.rawMemoryId,
    createdAt: record.createdAt
  });
}

function toMemoryClusterRow(record: MemoryClusterRecord): GacLanceRow {
  return toGacRow(record.id, record, {
    tenantId: record.tenantId,
    cellId: record.cellId,
    ...(record.sessionId ? { sessionId: record.sessionId } : {}),
    rawMemoryId: record.rawMemoryIds[0] ?? "",
    createdAt: record.createdAt
  });
}

function toClusterMetricRow(record: ClusterMetricRecord): GacLanceRow {
  return toGacRow(record.id, record, {
    tenantId: record.tenantId,
    cellId: record.cellId,
    ...(record.sessionId ? { sessionId: record.sessionId } : {}),
    createdAt: record.computedAt
  });
}

function toRepresentativeRow(record: MemoryRepresentativeRecord): GacLanceRow {
  return toGacRow(record.id, record, {
    tenantId: record.tenantId,
    cellId: record.cellId,
    ...(record.sessionId ? { sessionId: record.sessionId } : {}),
    rawMemoryId: record.sourceRawMemoryId ?? "",
    representativeId: record.id,
    createdAt: record.createdAt
  });
}

function toLineageRow(record: MemoryLineageRecord): GacLanceRow {
  return toGacRow(`${record.representativeId}:${record.rawMemoryId}`, record, {
    tenantId: record.tenantId,
    cellId: record.cellId,
    ...(record.sessionId ? { sessionId: record.sessionId } : {}),
    rawMemoryId: record.rawMemoryId,
    representativeId: record.representativeId,
    createdAt: record.createdAt
  });
}

function toConsolidationRunRow(record: ConsolidationRunRecord): GacLanceRow {
  return toGacRow(record.id, record, {
    tenantId: record.tenantId,
    cellId: record.cellId,
    ...(record.sessionId ? { sessionId: record.sessionId } : {}),
    createdAt: record.startedAt
  });
}

function toRetrievalAuditRow(record: RetrievalAuditRecord): GacLanceRow {
  return toGacRow(record.id, record, {
    tenantId: record.tenantId,
    cellId: record.cellId,
    ...(record.sessionId ? { sessionId: record.sessionId } : {}),
    rawMemoryId: record.expectedRawMemoryId,
    createdAt: record.createdAt
  });
}

function toModelMemoryActionRow(record: ModelMemoryActionRecord): GacLanceRow {
  return toGacRow(record.id, record, {
    tenantId: record.tenantId,
    cellId: record.cellId,
    sessionId: record.sessionId,
    rawMemoryId: record.targetIds[0] ?? "",
    createdAt: record.createdAt
  });
}

function toMemoryContradictionRow(record: MemoryContradictionRecord): GacLanceRow {
  return toGacRow(record.id, record, {
    tenantId: record.tenantId,
    cellId: record.cellId,
    ...(record.sessionId ? { sessionId: record.sessionId } : {}),
    rawMemoryId: record.rawMemoryIds[0] ?? "",
    createdAt: record.createdAt
  });
}

function toSourceDocumentRow(record: SourceDocumentRecord): GacLanceRow {
  return toGacRow(record.id, record, {
    tenantId: record.tenantId,
    cellId: record.cellId,
    ...(record.sessionId ? { sessionId: record.sessionId } : {}),
    createdAt: record.createdAt
  });
}

function toTrainingExampleRow(record: TrainingExampleRecord): GacLanceRow {
  return toGacRow(record.id, record, {
    tenantId: record.tenantId,
    cellId: record.cellId,
    ...(record.sessionId ? { sessionId: record.sessionId } : {}),
    rawMemoryId: record.sourceRawMemoryIds[0] ?? "",
    createdAt: record.createdAt
  });
}

function toContextPackTraceRow(record: ContextPackTraceRecord): GacLanceRow {
  return toGacRow(record.id, record, {
    tenantId: record.tenantId,
    cellId: record.cellId,
    sessionId: record.sessionId,
    contextPackId: record.contextPackId,
    createdAt: record.createdAt
  });
}

function toGacRow<T>(
  id: string,
  record: T,
  indexes: Partial<Omit<GacLanceRow, "id" | "gacTable" | "recordJson">> & { createdAt: string }
): GacLanceRow {
  return {
    id,
    gacTable: "",
    tenantId: "",
    cellId: "",
    sessionId: "",
    rawMemoryId: "",
    representativeId: "",
    contextPackId: "",
    ...indexes,
    recordJson: JSON.stringify(record)
  };
}

function assertRepresentativeLineage(records: MemoryRepresentativeRecord[], lineage: MemoryLineageRecord[]): void {
  const lineageIds = new Set(lineage.map((record) => record.representativeId));
  const missing = records.find((record) => (record.modelVisible || record.factual) && !lineageIds.has(record.id));
  if (missing) {
    throw new Error(`MISSING_LINEAGE: representative ${missing.id} is model-visible or factual and requires raw-memory lineage.`);
  }
}

function gacRowReferencesRawMemory(row: GacLanceRow, rawMemoryIds: Set<string>, decodeRecordJson: (value: string) => string): boolean {
  if (rawMemoryIds.size === 0) return false;
  if (rawMemoryIds.has(row.rawMemoryId) || rawMemoryIds.has(row.id)) return true;
  const record = safeParseJson<Record<string, unknown>>(decodeRecordJson(row.recordJson), {});
  return recordReferencesAny(record, rawMemoryIds, [
    "id",
    "rawMemoryId",
    "sourceRawMemoryId",
    "expectedRawMemoryId",
    "rawMemoryIds",
    "retrievedRawMemoryIds",
    "includedMemoryIds",
    "sourceRawMemoryIds"
  ]);
}

function gacRowReferencesRepresentative(row: GacLanceRow, representativeIds: Set<string>, decodeRecordJson: (value: string) => string): boolean {
  if (representativeIds.size === 0) return false;
  if (representativeIds.has(row.representativeId) || representativeIds.has(row.id)) return true;
  const record = safeParseJson<Record<string, unknown>>(decodeRecordJson(row.recordJson), {});
  return recordReferencesAny(record, representativeIds, [
    "id",
    "representativeId",
    "representativeIds",
    "retrievedRepresentativeIds",
    "affectedRepresentativeIds",
    "includedMemoryIds"
  ]);
}

function recordReferencesAny(record: Record<string, unknown>, ids: Set<string>, keys: string[]): boolean {
  return keys.some((key) => {
    const value = record[key];
    if (typeof value === "string") return ids.has(value);
    if (Array.isArray(value)) return value.some((item) => typeof item === "string" && ids.has(item));
    return false;
  });
}

function getStringField(record: Record<string, unknown>, key: string): string | undefined {
  const value = record[key];
  return typeof value === "string" ? value : undefined;
}

function buildGacFilters(options: GacListOptions): string[] {
  const filters: string[] = [];
  if (options.tenantId) filters.push(`tenantId = '${escapeSqlString(options.tenantId)}'`);
  if (options.cellId) filters.push(`cellId = '${escapeSqlString(options.cellId)}'`);
  if (options.sessionId) filters.push(`sessionId = '${escapeSqlString(options.sessionId)}'`);
  if (options.rawMemoryId) filters.push(`rawMemoryId = '${escapeSqlString(options.rawMemoryId)}'`);
  if (options.representativeId) filters.push(`representativeId = '${escapeSqlString(options.representativeId)}'`);
  if (options.contextPackId) filters.push(`contextPackId = '${escapeSqlString(options.contextPackId)}'`);
  return filters;
}

function buildRuntimeTraceFilters(options: RuntimeTraceListOptions): string[] {
  const filters: string[] = [];
  if (options.tenantId) filters.push(`tenantId = '${escapeSqlString(options.tenantId)}'`);
  if (options.cellId) filters.push(`cellId = '${escapeSqlString(options.cellId)}'`);
  if (options.sessionId) filters.push(`sessionId = '${escapeSqlString(options.sessionId)}'`);
  return filters;
}

function buildRuntimeTraceIdPredicate(row: RuntimeTraceLanceRow): string {
  const filters = [`traceId = '${escapeSqlString(row.traceId)}'`];
  if (row.tenantId) filters.push(`tenantId = '${escapeSqlString(row.tenantId)}'`);
  if (row.cellId) filters.push(`cellId = '${escapeSqlString(row.cellId)}'`);
  return filters.join(" AND ");
}

function makeWriteTraceId(tableName: string): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `gac_${tableName}_${crypto.randomUUID()}`;
  }
  return `gac_${tableName}_${Date.now()}`;
}

function escapeSqlString(value: string): string {
  return value.replace(/'/g, "''");
}

function parseFixedVectorDimension(type: string): number | undefined {
  const match = type.match(/FixedSizeList\[(\d+)\]/);
  return match?.[1] ? Number(match[1]) : undefined;
}
