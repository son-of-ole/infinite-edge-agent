import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  buildHeadlessWebGpuCiParityArtifact,
  buildMarkdownSummary,
  parseHeadlessWebGpuCiParityArgs,
  writeHeadlessWebGpuCiParityArtifact,
  type HeadlessWebGpuCiParityProbeResult,
} from "./headlessWebGpuCiParity";

const passingProbe: HeadlessWebGpuCiParityProbeResult = {
  webGpuAvailable: true,
  adapterFeatures: ["shader-f16"],
  adapterInfo: {
    vendor: "Apple",
    architecture: "apple",
    device: "Apple M-series GPU",
    description: "Apple M-series GPU",
    isFallbackAdapter: false,
  },
  limits: {
    maxStorageBufferBindingSize: 1024,
  },
  expectedOutput: [2, 4, 6, 8],
  actualOutput: [2, 4, 6, 8],
  durationMs: 12.5,
};

describe("headless WebGPU CI parity artifact", () => {
  it("passes browser-runner parity only when a real WebGPU compute probe matches", () => {
    const artifact = buildHeadlessWebGpuCiParityArtifact({
      createdAt: "2026-05-20T00:00:00.000Z",
      lane: "browser-runner",
      required: true,
      browser: { engine: "chromium", channel: "chrome" },
      probe: passingProbe,
    });

    expect(artifact).toMatchObject({
      name: "headless-webgpu-ci-parity",
      lane: "browser-runner",
      status: "passed",
      passed: true,
      required: true,
      webGpuAvailable: true,
      probe: {
        expectedOutput: [2, 4, 6, 8],
        actualOutput: [2, 4, 6, 8],
        outputMatches: true,
      },
      summary: {
        lane: "browser-runner",
        status: "passed",
        webGpuAvailable: true,
        outputMatches: true,
        realHardwareAdapter: true,
      },
    });
  });

  it("records optional browser-runner WebGPU absence as an honest skip instead of a pass", () => {
    const artifact = buildHeadlessWebGpuCiParityArtifact({
      createdAt: "2026-05-20T00:00:00.000Z",
      lane: "browser-runner",
      required: false,
      probe: {
        webGpuAvailable: false,
        adapterFeatures: [],
        expectedOutput: [2, 4, 6, 8],
        actualOutput: [],
        error: "navigator.gpu is unavailable",
      },
    });

    expect(artifact.status).toBe("skipped");
    expect(artifact.passed).toBe(false);
    expect(artifact.failureReason).toContain("navigator.gpu is unavailable");
    expect(artifact.summary.outputMatches).toBe(false);
  });

  it("fails required browser-runner parity when WebGPU is unavailable", () => {
    const artifact = buildHeadlessWebGpuCiParityArtifact({
      createdAt: "2026-05-20T00:00:00.000Z",
      lane: "browser-runner",
      required: true,
      probe: {
        webGpuAvailable: false,
        adapterFeatures: [],
        expectedOutput: [2, 4, 6, 8],
        actualOutput: [],
        error: "requestAdapter returned null",
      },
    });

    expect(artifact.status).toBe("failed");
    expect(artifact.passed).toBe(false);
    expect(artifact.failureReason).toContain("requestAdapter returned null");
  });

  it("fails required browser-runner parity when the compute shader output mismatches", () => {
    const artifact = buildHeadlessWebGpuCiParityArtifact({
      createdAt: "2026-05-20T00:00:00.000Z",
      lane: "browser-runner",
      required: true,
      probe: {
        ...passingProbe,
        webGpuAvailable: true,
        expectedOutput: [2, 4, 6, 8],
        actualOutput: [2, 4, 6, 7],
        durationMs: 10,
      },
    });

    expect(artifact.status).toBe("failed");
    expect(artifact.passed).toBe(false);
    expect(artifact.probe.outputMatches).toBe(false);
    expect(artifact.failureReason).toContain("compute shader output did not match");
  });

  it("reports optional compute mismatch as failed because the probe ran and was wrong", () => {
    const artifact = buildHeadlessWebGpuCiParityArtifact({
      createdAt: "2026-05-20T00:00:00.000Z",
      lane: "browser-runner",
      required: false,
      probe: {
        ...passingProbe,
        actualOutput: [2, 4, 6, 7],
      },
    });

    expect(artifact.status).toBe("failed");
    expect(artifact.passed).toBe(false);
    expect(artifact.failureReason).toContain("compute shader output did not match");
  });

  it("rejects software WebGPU adapters even when the compute shader output matches", () => {
    const artifact = buildHeadlessWebGpuCiParityArtifact({
      createdAt: "2026-05-20T00:00:00.000Z",
      lane: "browser-runner",
      required: true,
      probe: {
        ...passingProbe,
        adapterInfo: {
          vendor: "Google",
          architecture: "swiftshader",
          device: "SwiftShader Device",
          description: "SwiftShader Vulkan software adapter",
          isFallbackAdapter: true,
        },
      },
    });

    expect(artifact.status).toBe("failed");
    expect(artifact.passed).toBe(false);
    expect(artifact.summary.realHardwareAdapter).toBe(false);
    expect(artifact.failureReason).toContain("software WebGPU adapter");
  });

  it("does not pass when adapter identity is unavailable because real hardware cannot be proven", () => {
    const artifact = buildHeadlessWebGpuCiParityArtifact({
      createdAt: "2026-05-20T00:00:00.000Z",
      lane: "browser-runner",
      required: true,
      probe: {
        ...passingProbe,
        adapterInfo: undefined,
      },
    });

    expect(artifact.status).toBe("failed");
    expect(artifact.passed).toBe(false);
    expect(artifact.failureReason).toContain("adapter identity is unavailable");
  });

  it("does not pass when adapter identity fields are literal unknown placeholders", () => {
    const artifact = buildHeadlessWebGpuCiParityArtifact({
      createdAt: "2026-05-20T00:00:00.000Z",
      lane: "browser-runner",
      required: true,
      probe: {
        ...passingProbe,
        adapterInfo: {
          vendor: "unknown",
          architecture: "unknown",
          device: "unknown",
          description: "unknown",
          isFallbackAdapter: false,
        },
      },
    });

    expect(artifact.status).toBe("failed");
    expect(artifact.passed).toBe(false);
    expect(artifact.summary.adapterIdentityKnown).toBe(false);
    expect(artifact.failureReason).toContain("adapter identity is unavailable");
  });

  it("rejects non-browser lanes so only real browser WebGPU can pass parity", () => {
    expect(() => parseHeadlessWebGpuCiParityArgs([
      "--lane",
      "dawn-lavapipe",
    ], {})).toThrow(/Only the real browser-runner lane is supported/);
  });

  it("parses CLI and env controls for a real browser-runner lane by default", () => {
    expect(parseHeadlessWebGpuCiParityArgs([], {})).toMatchObject({
      lane: "browser-runner",
      required: false,
      artifactDir: ".artifacts/evals",
    });

    expect(parseHeadlessWebGpuCiParityArgs([
      "--required",
      "--artifact-dir",
      ".artifacts/custom",
    ], {})).toMatchObject({
      lane: "browser-runner",
      required: true,
      artifactDir: ".artifacts/custom",
    });

    expect(parseHeadlessWebGpuCiParityArgs([], {
      HEADLESS_WEBGPU_CI_REQUIRED: "true",
      HEADLESS_WEBGPU_CI_LANE: "browser-runner",
      HEADLESS_WEBGPU_CI_BROWSER_CHANNEL: "chrome",
    })).toMatchObject({
      lane: "browser-runner",
      required: true,
      browserChannel: "chrome",
    });
  });

  it("writes timestamped and latest artifacts plus a markdown summary", async () => {
    const artifactDir = await mkdtemp(resolve(tmpdir(), "headless-webgpu-ci-"));
    const artifact = buildHeadlessWebGpuCiParityArtifact({
      createdAt: "2026-05-20T00:00:00.000Z",
      lane: "browser-runner",
      required: true,
      probe: passingProbe,
    });

    const paths = await writeHeadlessWebGpuCiParityArtifact(artifact, { artifactDir });
    const latest = JSON.parse(await readFile(paths.latestPath, "utf8")) as typeof artifact;
    const summary = await readFile(paths.summaryPath, "utf8");

    expect(latest.name).toBe("headless-webgpu-ci-parity");
    expect(latest.status).toBe("passed");
    expect(summary).toBe(buildMarkdownSummary(artifact));
    expect(paths.resultsPath).toContain("headless-webgpu-ci-parity");
  });
});
