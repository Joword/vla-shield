import { create } from "zustand";

export interface LatencyBreakdown {
  ingest_ms: number;
  urdf_fk_ms: number | null;
  physics_ms: number;
  collision_ms: number;
  tf2_ms: number | null;
  arbiter_ms: number;
  shadow_ms: number | null;
  total_ms: number;
}

export interface TelemetryState {
  risk: number;
  decision: string;
  ontologyIds: string[];
  /** Optional per-ontology detail strings (e.g. joint=... from arbiter). */
  ontologyDetails: Record<string, string>;
  sceneRev: number;
  tsNs: number;
  latencyBreakdown: LatencyBreakdown | null;
  history: { ts: number; risk: number }[];
  latencyHistory: { ts: number; total_ms: number }[];
  update: (msg: TelemetryMessage) => void;
}

interface TelemetryMessage {
  risk: number;
  decision: string;
  ontology_ids: string[];
  ontology_details?: Record<string, string>;
  scene_rev: number;
  ts_ns: number;
  latency?: LatencyBreakdown;
}

const MAX_HISTORY = 300;

export const useTelemetryStore = create<TelemetryState>((set) => ({
  risk: 0,
  decision: "PASS",
  ontologyIds: [],
  ontologyDetails: {},
  sceneRev: 0,
  tsNs: 0,
  latencyBreakdown: null,
  history: [],
  latencyHistory: [],
  update: (msg) =>
    set((state) => {
      const entry = { ts: msg.ts_ns, risk: msg.risk };
      const latEntry = msg.latency
        ? { ts: msg.ts_ns, total_ms: msg.latency.total_ms }
        : null;

      const history =
        state.history.length >= MAX_HISTORY
          ? [...state.history.slice(1), entry]
          : [...state.history, entry];

      const latencyHistory = latEntry
        ? state.latencyHistory.length >= MAX_HISTORY
          ? [...state.latencyHistory.slice(1), latEntry]
          : [...state.latencyHistory, latEntry]
        : state.latencyHistory;

      return {
        risk: msg.risk,
        decision: msg.decision,
        ontologyIds: msg.ontology_ids,
        ontologyDetails: msg.ontology_details ?? {},
        sceneRev: msg.scene_rev,
        tsNs: msg.ts_ns,
        latencyBreakdown: msg.latency ?? state.latencyBreakdown,
        history,
        latencyHistory,
      };
    }),
}));
