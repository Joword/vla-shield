import { create } from "zustand";

export interface TelemetryState {
  risk: number;
  decision: string;
  ontologyIds: string[];
  /** Optional per-ontology detail strings (e.g. joint=... from arbiter). */
  ontologyDetails: Record<string, string>;
  sceneRev: number;
  tsNs: number;
  history: { ts: number; risk: number }[];
  update: (msg: TelemetryMessage) => void;
}

interface TelemetryMessage {
  risk: number;
  decision: string;
  ontology_ids: string[];
  ontology_details?: Record<string, string>;
  scene_rev: number;
  ts_ns: number;
}

const MAX_HISTORY = 300;

export const useTelemetryStore = create<TelemetryState>((set) => ({
  risk: 0,
  decision: "PASS",
  ontologyIds: [],
  ontologyDetails: {},
  sceneRev: 0,
  tsNs: 0,
  history: [],
  update: (msg) =>
    set((state) => {
      const entry = { ts: msg.ts_ns, risk: msg.risk };
      const history =
        state.history.length >= MAX_HISTORY
          ? [...state.history.slice(1), entry]
          : [...state.history, entry];
      return {
        risk: msg.risk,
        decision: msg.decision,
        ontologyIds: msg.ontology_ids,
        ontologyDetails: msg.ontology_details ?? {},
        sceneRev: msg.scene_rev,
        tsNs: msg.ts_ns,
        history,
      };
    }),
}));
