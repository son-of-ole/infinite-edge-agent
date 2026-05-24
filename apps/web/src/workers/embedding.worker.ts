import { env, pipeline } from "@huggingface/transformers";

interface InitMessage {
  id: string;
  type: "init";
  modelId: string;
  preferWebGPU: boolean;
}

interface EmbedMessage {
  id: string;
  type: "embed";
  texts: string[];
}

type IncomingMessage = InitMessage | EmbedMessage;

type EmbeddingDevice = "webgpu" | "wasm";
type PipelineLoadOptions = {
  device: EmbeddingDevice;
  dtype: "q8";
};
type EmbeddingOptions = {
  pooling: "mean";
  normalize: true;
};
type FeatureExtractor = (texts: string | string[], options: Record<string, unknown>) => Promise<{
  data: Float32Array | number[];
  dims?: number[];
  tolist?: () => number[] | number[][];
}>;
const createFeatureExtractor = pipeline as (
  task: "feature-extraction",
  modelId: string,
  options: PipelineLoadOptions
) => Promise<FeatureExtractor>;

let extractor: FeatureExtractor | null = null;
let loadedModelId: string | null = null;

// Keep remote downloads enabled, but rely on the browser cache after first load.
env.allowLocalModels = false;

async function init(modelId: string, preferWebGPU: boolean): Promise<{ device: EmbeddingDevice; modelId: string }> {
  if (extractor && loadedModelId === modelId) {
    return { device: preferWebGPU ? "webgpu" : "wasm", modelId };
  }

  const preferredDevice: EmbeddingDevice = preferWebGPU ? "webgpu" : "wasm";
  try {
    extractor = await createFeatureExtractor("feature-extraction", modelId, {
      device: preferredDevice,
      dtype: "q8"
    });
    loadedModelId = modelId;
    return { device: preferredDevice, modelId };
  } catch (error) {
    if (preferredDevice === "webgpu") {
      extractor = await createFeatureExtractor("feature-extraction", modelId, {
        device: "wasm",
        dtype: "q8"
      });
      loadedModelId = modelId;
      return { device: "wasm", modelId };
    }
    throw error;
  }
}

function tensorToVectors(output: Awaited<ReturnType<FeatureExtractor>>, batchSize: number): number[][] {
  if (typeof output.tolist === "function") {
    const list = output.tolist();
    if (Array.isArray(list[0])) return list as number[][];
    return [list as number[]];
  }

  const data = Array.from(output.data ?? []);
  const dims = output.dims;
  const dim = dims && dims.length > 1 ? dims[dims.length - 1] ?? data.length : Math.floor(data.length / batchSize);
  const vectors: number[][] = [];
  for (let i = 0; i < batchSize; i += 1) {
    vectors.push(data.slice(i * dim, (i + 1) * dim));
  }
  return vectors;
}

async function embed(texts: string[]): Promise<number[][]> {
  if (!extractor) throw new Error("Embedding pipeline is not initialized.");
  const options: EmbeddingOptions = { pooling: "mean", normalize: true };
  const output = await extractor(texts, options);
  return tensorToVectors(output, texts.length);
}

self.onmessage = async (event: MessageEvent<IncomingMessage>) => {
  const message = event.data;
  try {
    if (message.type === "init") {
      const result = await init(message.modelId, message.preferWebGPU);
      self.postMessage({ id: message.id, type: "init:ok", result });
      return;
    }

    if (message.type === "embed") {
      const vectors = await embed(message.texts);
      self.postMessage({ id: message.id, type: "embed:ok", vectors });
    }
  } catch (error) {
    self.postMessage({
      id: message.id,
      type: "error",
      error: error instanceof Error ? error.message : String(error)
    });
  }
};
