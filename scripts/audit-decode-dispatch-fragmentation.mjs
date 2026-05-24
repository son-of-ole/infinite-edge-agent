#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";

const inputs = process.argv.slice(2);
if (inputs.length === 0) {
  console.error("Usage: node scripts/audit-decode-dispatch-fragmentation.mjs <file-or-dir> [...]");
  process.exit(2);
}

const files = [];
for (const input of inputs) collect(input, files);

const patterns = [
  [/queue\.submit\s*\(/g, "queue.submit"],
  [/onSubmittedWorkDone\s*\(/g, "onSubmittedWorkDone"],
  [/createCommandEncoder\s*\(/g, "createCommandEncoder"],
  [/beginComputePass\s*\(/g, "beginComputePass"],
  [/dispatchWorkgroups\s*\(/g, "dispatchWorkgroups"],
  [/mapAsync\s*\(/g, "mapAsync"],
  [/readWebGpuResidentTensor/g, "readWebGpuResidentTensor"],
];

let totalFindings = 0;
for (const file of files) {
  const text = fs.readFileSync(file, "utf8");
  const fileFindings = [];
  for (const [regex, label] of patterns) {
    const matches = [...text.matchAll(regex)];
    if (matches.length > 0) fileFindings.push({ label, count: matches.length });
  }
  if (fileFindings.length > 0) {
    totalFindings += fileFindings.reduce((sum, item) => sum + item.count, 0);
    console.log(`\n${file}`);
    for (const item of fileFindings) console.log(`  ${item.label}: ${item.count}`);
  }
}

console.log(`\nTotal dispatch-fragmentation findings: ${totalFindings}`);

function collect(target, out) {
  if (!fs.existsSync(target)) return;
  const stat = fs.statSync(target);
  if (stat.isDirectory()) {
    for (const entry of fs.readdirSync(target)) collect(path.join(target, entry), out);
    return;
  }
  if (/\.(ts|tsx|js|mjs|wgsl)$/.test(target)) out.push(target);
}
