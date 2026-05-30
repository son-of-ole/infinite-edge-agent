import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { materializeHostedBenchmarkArtifact } from "./materializeHostedBenchmarkArtifact";

function makeArtifact() {
  return {
    name: "browser-preview-benchmark",
    passed: true,
    summary: {
      runtimeBackendId: "compiled-browser-webllm",
      productionDeployReadyPassed: true,
    },
  };
}

describe("hosted benchmark artifact materializer", () => {
  it("writes inline hosted benchmark JSON to a stable output path", async () => {
    const dir = await mkdtemp(join(tmpdir(), "hosted-benchmark-materialize-"));
    const outputPath = join(dir, "browser-runtime-bench-latest.json");

    const result = await materializeHostedBenchmarkArtifact({
      inlineJson: JSON.stringify(makeArtifact()),
      outputPath,
    });

    expect(result).toMatchObject({
      artifactPath: outputPath,
      source: "inline_json",
    });
    expect(result.bytes).toBeGreaterThan(0);

    const written = JSON.parse(await readFile(outputPath, "utf8")) as ReturnType<typeof makeArtifact>;
    expect(written.name).toBe("browser-preview-benchmark");
    expect(written.summary.runtimeBackendId).toBe("compiled-browser-webllm");
  });

  it("writes base64 hosted benchmark JSON to a stable output path", async () => {
    const dir = await mkdtemp(join(tmpdir(), "hosted-benchmark-materialize-"));
    const outputPath = join(dir, "browser-runtime-bench-latest.json");
    const base64Json = Buffer.from(JSON.stringify(makeArtifact()), "utf8").toString("base64");

    const result = await materializeHostedBenchmarkArtifact({
      base64Json,
      outputPath,
    });

    expect(result).toMatchObject({
      artifactPath: outputPath,
      source: "base64_json",
    });

    const written = JSON.parse(await readFile(outputPath, "utf8")) as ReturnType<typeof makeArtifact>;
    expect(written.summary.runtimeBackendId).toBe("compiled-browser-webllm");
  });

  it("rejects ambiguous hosted benchmark artifact sources", async () => {
    const dir = await mkdtemp(join(tmpdir(), "hosted-benchmark-materialize-"));
    const outputPath = join(dir, "browser-runtime-bench-latest.json");
    const base64Json = Buffer.from(JSON.stringify({
      ...makeArtifact(),
      summary: { runtimeBackendId: "wrong-backend" },
    }), "utf8").toString("base64");

    await expect(materializeHostedBenchmarkArtifact({
      inlineJson: JSON.stringify(makeArtifact()),
      base64Json,
      outputPath,
    })).rejects.toThrow("Provide exactly one hosted benchmark artifact source.");
  });

  it("rejects missing hosted benchmark input", async () => {
    await expect(materializeHostedBenchmarkArtifact({
      outputPath: join(await mkdtemp(join(tmpdir(), "hosted-benchmark-materialize-")), "artifact.json"),
    })).rejects.toThrow("Provide HOSTED_BENCHMARK_ARTIFACT_JSON, HOSTED_BENCHMARK_ARTIFACT_BASE64, or HOSTED_BENCHMARK_ARTIFACT_URL.");
  });

  it("rejects non-json inline artifact input", async () => {
    await expect(materializeHostedBenchmarkArtifact({
      inlineJson: "not json",
      outputPath: join(await mkdtemp(join(tmpdir(), "hosted-benchmark-materialize-")), "artifact.json"),
    })).rejects.toThrow("Hosted benchmark artifact JSON is invalid.");
  });
});
