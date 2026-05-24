import react from "@vitejs/plugin-react";
import { existsSync, readdirSync, rmSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { defineConfig, loadEnv, type Plugin } from "vite";

export default defineConfig(({ mode }) => {
  const env = { ...loadEnv(mode, process.cwd(), ""), ...process.env };
  return {
    plugins: [react(), omitLocalModelArtifactsFromBuild(env)],
    build: {
      chunkSizeWarningLimit: 6500,
      rollupOptions: {
        output: {
          manualChunks(id) {
            if (id.includes("@huggingface/transformers") || id.includes("onnxruntime")) return "embedding-runtime";
            if (id.includes("node_modules/react")) return "react-vendor";
          }
        }
      }
    },
    worker: {
      format: "es"
    },
    optimizeDeps: {
      exclude: ["@huggingface/transformers"]
    },
    server: {
      headers: {
        // Enables SharedArrayBuffer where supported. Useful for WASM-backed ML runtimes.
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp"
      }
    }
  };
});

function omitLocalModelArtifactsFromBuild(env: Record<string, string | undefined>): Plugin {
  return {
    name: "omit-local-model-artifacts-from-build",
    closeBundle() {
      const modelDir = fileURLToPath(new URL("./dist/models/", import.meta.url));
      if (!existsSync(modelDir)) return;
      omitModelArtifacts(modelDir, env);
    }
  };
}

function omitModelArtifacts(dir: string, env: Record<string, string | undefined>): void {
  for (const entry of readdirSync(dir)) {
    const path = `${dir}/${entry}`;
    if (statSync(path).isDirectory()) {
      omitModelArtifacts(path, env);
      continue;
    }
    const isOpaqueModelArtifact = entry.endsWith(".litertlm") || entry.endsWith(".litertlm.sha256");
    const isUnlockedShard = /\.(bin|safetensors|onnx|npy|npz|gguf)$/i.test(entry);
    if (isOpaqueModelArtifact || (isUnlockedShard && env.VITE_BUNDLE_UNLOCKED_MODEL !== "true")) {
      rmSync(path, { force: true });
    }
  }
}
