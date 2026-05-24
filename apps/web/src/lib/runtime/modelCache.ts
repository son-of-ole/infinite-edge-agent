export interface BrowserModelCacheEntry {
  kind: "cache-storage" | "indexeddb";
  name: string;
  modelRelated: boolean;
}

export interface BrowserModelCacheSnapshot {
  usageBytes?: number;
  quotaBytes?: number;
  entries: BrowserModelCacheEntry[];
  checkedAt: string;
}

const MODEL_CACHE_PATTERN = /(qwen|unlocked|mlc|model|transformers|huggingface|onnx|wasm)/i;

export function isLikelyModelCacheName(name: string): boolean {
  return MODEL_CACHE_PATTERN.test(name);
}

export async function inspectBrowserModelCache(): Promise<BrowserModelCacheSnapshot> {
  const entries: BrowserModelCacheEntry[] = [];
  const storageEstimate = await getStorageEstimate();

  if ("caches" in globalThis) {
    const keys = await caches.keys();
    entries.push(...keys.map((name) => ({
      kind: "cache-storage" as const,
      name,
      modelRelated: isLikelyModelCacheName(name),
    })));
  }

  const databaseNames = await listIndexedDbNames();
  entries.push(...databaseNames.map((name) => ({
    kind: "indexeddb" as const,
    name,
    modelRelated: isLikelyModelCacheName(name),
  })));

  return {
    ...(storageEstimate.usage !== undefined ? { usageBytes: storageEstimate.usage } : {}),
    ...(storageEstimate.quota !== undefined ? { quotaBytes: storageEstimate.quota } : {}),
    entries: entries.sort((a, b) => Number(b.modelRelated) - Number(a.modelRelated) || a.name.localeCompare(b.name)),
    checkedAt: new Date().toISOString(),
  };
}

export async function clearBrowserModelCaches(): Promise<BrowserModelCacheEntry[]> {
  const snapshot = await inspectBrowserModelCache();
  const targets = snapshot.entries.filter((entry) => entry.modelRelated);

  await Promise.all(targets.map(async (entry) => {
    if (entry.kind === "cache-storage" && "caches" in globalThis) {
      await caches.delete(entry.name);
    }
    if (entry.kind === "indexeddb" && "indexedDB" in globalThis) {
      await deleteIndexedDb(entry.name);
    }
  }));

  return targets;
}

async function getStorageEstimate(): Promise<StorageEstimate> {
  if (!("navigator" in globalThis) || !navigator.storage?.estimate) return {};
  try {
    return await navigator.storage.estimate();
  } catch {
    return {};
  }
}

async function listIndexedDbNames(): Promise<string[]> {
  if (!("indexedDB" in globalThis) || typeof indexedDB.databases !== "function") return [];
  try {
    const databases = await indexedDB.databases();
    return databases.map((database) => database.name).filter((name): name is string => Boolean(name));
  } catch {
    return [];
  }
}

function deleteIndexedDb(name: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.deleteDatabase(name);
    request.onsuccess = () => resolve();
    request.onblocked = () => resolve();
    request.onerror = () => reject(request.error ?? new Error(`Failed to delete IndexedDB database: ${name}`));
  });
}
