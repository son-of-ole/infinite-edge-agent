import { listLocalModelOptionsFromRegistry, type LocalModelOptionFromRegistry } from "./modelRegistry";

export interface LocalModelOption {
  id: string;
  label: string;
  backend: LocalModelOptionFromRegistry["backend"];
  backendLabel: string;
  productionRole: LocalModelOptionFromRegistry["productionRole"];
  deployReadyCandidate: boolean;
  notes: string;
  estimatedDownload: string;
  modelAssetPath?: string;
}

export const LOCAL_MODEL_OPTIONS: LocalModelOption[] = listLocalModelOptionsFromRegistry();
