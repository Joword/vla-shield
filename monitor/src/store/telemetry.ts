import { create } from "zustand";

export type WsStatus = "connecting" | "connected" | "reconnecting" | "disconnected";

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

export type Vec3 = [number, number, number];

export interface SceneZone {
  min: Vec3;
  max: Vec3;
  label: string;
  ontology_id: string;
}

export interface HistoryEntry {
  ts: number;
  risk: number;
  decision: string;
  ontologyIds: string[];
}

export interface TelemetryState {
  wsStatus: WsStatus;
  risk: number;
  decision: string;
  ontologyIds: string[];
  ontologyDetails: Record<string, string>;
  sceneRev: number;
  tsNs: number;
  currentJoints: number[];
  projectedJoints: number[];
  skeleton: Vec3[];
  shadowPath: Vec3[];
  ee: Vec3 | null;
  zones: SceneZone[];
  latencyBreakdown: LatencyBreakdown | null;
  history: HistoryEntry[];
  latencyHistory: { ts: number; total_ms: number }[];
  setWsStatus: (status: WsStatus) => void;
  update: (msg: TelemetryMessage) => void;
}

export interface TelemetryMessage {
  risk: number;
  decision: string;
  ontology_ids: string[];
  ontology_details?: Record<string, string>;
  scene_rev: number;
  ts_ns: number;
  latency?: LatencyBreakdown;
  current_joints?: number[];
  projected_joints?: number[];
  skeleton?: number[][];
  shadow_path?: number[][];
  ee?: number[] | null;
  zones?: SceneZone[];
}

const MAX_HISTORY = 300;

function asVec3(p: number[] | undefined | null): Vec3 | null {
  if (!p || p.length < 3) return null;
  return [p[0], p[1], p[2]];
}

function asVec3List(pts: number[][] | undefined): Vec3[] {
  if (!pts) return [];
  return pts
    .map((p) => asVec3(p))
    .filter((p): p is Vec3 => p !== null);
}

export const useTelemetryStore = create<TelemetryState>((set) => ({
  wsStatus: "disconnected",
  risk: 0,
  decision: "PASS",
  ontologyIds: [],
  ontologyDetails: {},
  sceneRev: 0,
  tsNs: 0,
  currentJoints: [],
  projectedJoints: [],
  skeleton: [],
  shadowPath: [],
  ee: null,
  zones: [],
  latencyBreakdown: null,
  history: [],
  latencyHistory: [],
  setWsStatus: (wsStatus) => set({ wsStatus }),
  update: (msg) =>
    set((state) => {
      const entry: HistoryEntry = {
        ts: msg.ts_ns,
        risk: msg.risk,
        decision: msg.decision,
        ontologyIds: msg.ontology_ids ?? [],
      };
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
        currentJoints: msg.current_joints ?? state.currentJoints,
        projectedJoints: msg.projected_joints ?? state.projectedJoints,
        skeleton: msg.skeleton ? asVec3List(msg.skeleton) : state.skeleton,
        shadowPath: msg.shadow_path ? asVec3List(msg.shadow_path) : state.shadowPath,
        ee: msg.ee ? asVec3(msg.ee) : state.ee,
        zones: msg.zones ?? state.zones,
        latencyBreakdown: msg.latency ?? state.latencyBreakdown,
        history,
        latencyHistory,
      };
    }),
}));
