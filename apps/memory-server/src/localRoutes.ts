export interface LocalRouteExposureInput {
  host: string;
  value?: string;
}

export function shouldExposeLocalRoutes(input: LocalRouteExposureInput): boolean {
  const configured = input.value?.trim().toLowerCase();
  if (configured === "true" || configured === "1" || configured === "yes" || configured === "on") return true;
  if (configured === "false" || configured === "0" || configured === "no" || configured === "off") return false;
  return isLoopbackHost(input.host);
}

function isLoopbackHost(host: string): boolean {
  const normalized = host.trim().toLowerCase();
  return normalized === "localhost"
    || normalized === "127.0.0.1"
    || normalized === "::1"
    || normalized === "[::1]";
}
