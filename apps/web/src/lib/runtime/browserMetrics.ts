export interface BrowserMetric {
  name: string;
  valueMs: number;
  at: string;
}

export interface BrowserMetricSink {
  addMetric(name: string, valueMs: number): void;
}

export function makeBrowserMetric(name: string, valueMs: number, now = new Date()): BrowserMetric {
  return {
    name,
    valueMs: Math.max(0, Math.round(valueMs)),
    at: now.toISOString(),
  };
}

export async function timed<T>(
  name: string,
  sink: BrowserMetricSink | undefined,
  action: () => Promise<T>,
): Promise<T> {
  const started = performance.now();
  try {
    return await action();
  } finally {
    sink?.addMetric(name, performance.now() - started);
  }
}
