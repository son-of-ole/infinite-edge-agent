import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawn } from "node:child_process";

const packageDir = resolve("packages/sdk");
const workspaceRoot = process.cwd();
const tempRoot = await mkdtemp(join(tmpdir(), "infinite-edge-sdk-package-"));
const appDir = join(tempRoot, "consumer");

await mkdir(appDir, { recursive: true });
await writeFile(join(appDir, "package.json"), JSON.stringify({ type: "module", dependencies: {} }, null, 2));

const packResult = await run("npm", ["pack", packageDir, "--silent", "--pack-destination", tempRoot], workspaceRoot);
const tarballName = packResult.stdout.trim().split(/\r?\n/).filter(Boolean).at(-1);
if (!tarballName) {
  throw new Error("npm pack did not report a package tarball.");
}

await run("npm", ["install", "--silent", join(tempRoot, tarballName)], appDir);
const packageEntrypointResult = await run(
  process.execPath,
  [
    "--input-type=module",
    "-e",
    [
      "import { buildInfiniteEdgeAgentUrl, mountInfiniteEdgeAgent } from '@infinite-edge-agent/browser-sdk';",
      "if (typeof buildInfiniteEdgeAgentUrl !== 'function' || typeof mountInfiniteEdgeAgent !== 'function') throw new Error('missing SDK exports');",
      "const url = buildInfiniteEdgeAgentUrl({ agentUrl: 'https://agent.example.com/app', deployment: 'browser-only' });",
      "if (url.searchParams.get('deploymentPreset') !== 'browser-only') throw new Error('SDK package entrypoint did not build expected URL');",
      "console.log('SDK package entrypoint smoke: PASS');",
    ].join(" "),
  ],
  appDir,
);
if (packageEntrypointResult.stdout.trim()) {
  console.log(packageEntrypointResult.stdout.trim());
}

async function run(command, args, cwd) {
  return new Promise((resolveRun, reject) => {
    const child = spawn(command, args, { cwd, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolveRun({ stdout, stderr });
        return;
      }
      reject(new Error(`${command} ${args.join(" ")} failed with ${code}\n${stdout}\n${stderr}`));
    });
  });
}
