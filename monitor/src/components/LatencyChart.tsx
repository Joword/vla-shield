"use client";

import { useTelemetryStore, LatencyBreakdown } from "@/store/telemetry";

const STAGES: { key: keyof LatencyBreakdown; label: string; color: string }[] = [
  { key: "ingest_ms",    label: "Ingest",    color: "#6366f1" },
  { key: "urdf_fk_ms",  label: "URDF FK",   color: "#22d3ee" },
  { key: "physics_ms",  label: "Physics",   color: "#34d399" },
  { key: "collision_ms",label: "Collision",  color: "#fbbf24" },
  { key: "tf2_ms",      label: "tf2",       color: "#fb923c" },
  { key: "arbiter_ms",  label: "Arbiter",   color: "#f87171" },
  { key: "shadow_ms",   label: "Shadow",    color: "#a78bfa" },
];

const BUDGET_MS = 5.0;

function Segment({
  value,
  total,
  color,
  label,
}: {
  value: number;
  total: number;
  color: string;
  label: string;
}) {
  const pct = total > 0 ? (value / total) * 100 : 0;
  if (pct < 0.5) return null;
  return (
    <div
      className="relative flex items-center justify-center h-full overflow-hidden"
      style={{ width: `${pct}%`, backgroundColor: color }}
      title={`${label}: ${value.toFixed(2)} ms`}
    >
      {pct > 8 && (
        <span className="text-[10px] text-white font-mono whitespace-nowrap px-0.5">
          {value.toFixed(1)}
        </span>
      )}
    </div>
  );
}

export default function LatencyChart() {
  const latency = useTelemetryStore((s) => s.latencyBreakdown);
  const latencyHistory = useTelemetryStore((s) => s.latencyHistory);

  if (!latency) {
    return (
      <div className="p-4 border-b border-gray-700">
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-2">
          Latency Breakdown
        </h2>
        <p className="text-xs text-gray-500">Waiting for telemetry…</p>
      </div>
    );
  }

  const total = latency.total_ms;
  const overBudget = total > BUDGET_MS;

  return (
    <div className="p-4 border-b border-gray-700">
      <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Latency Breakdown
      </h2>

      {/* Stacked bar */}
      <div className="flex h-6 w-full rounded overflow-hidden mb-2">
        {STAGES.map(({ key, label, color }) => {
          const raw = latency[key];
          const val = typeof raw === "number" ? raw : 0;
          return (
            <Segment
              key={key}
              value={val}
              total={total}
              color={color}
              label={label}
            />
          );
        })}
      </div>

      {/* Total + budget indicator */}
      <div className="flex items-center justify-between text-xs mb-3">
        <span className={`font-mono font-semibold ${overBudget ? "text-danger" : "text-success"}`}>
          {total.toFixed(2)} ms total
        </span>
        <span className="text-gray-500">
          budget: {BUDGET_MS} ms
          {overBudget && <span className="text-danger ml-1">⚠ exceeded</span>}
        </span>
      </div>

      {/* Per-stage table */}
      <table className="w-full text-xs text-gray-400">
        <tbody>
          {STAGES.map(({ key, label, color }) => {
            const raw = latency[key];
            if (raw === null || raw === undefined) return null;
            return (
              <tr key={key} className="border-t border-gray-700/50">
                <td className="py-0.5 pr-2">
                  <span
                    className="inline-block w-2 h-2 rounded-sm mr-1"
                    style={{ backgroundColor: color }}
                  />
                  {label}
                </td>
                <td className="py-0.5 text-right font-mono text-gray-300">
                  {raw.toFixed(3)} ms
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {/* Mini latency sparkline (last N samples) */}
      {latencyHistory.length > 1 && (
        <div className="mt-3">
          <p className="text-xs text-gray-500 mb-1">Recent total_ms</p>
          <LatencySparkline data={latencyHistory.map((h) => h.total_ms)} />
        </div>
      )}
    </div>
  );
}

function LatencySparkline({ data }: { data: number[] }) {
  const max = Math.max(...data, BUDGET_MS);
  const height = 32;
  const width = 200;
  const pts = data
    .slice(-60)
    .map((v, i, arr) => {
      const x = (i / (arr.length - 1)) * width;
      const y = height - (v / max) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  const budgetY = height - (BUDGET_MS / max) * height;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      width="100%"
      height={height}
      className="overflow-visible"
    >
      <line
        x1={0}
        y1={budgetY}
        x2={width}
        y2={budgetY}
        stroke="#ef4444"
        strokeWidth={0.8}
        strokeDasharray="3 2"
      />
      <polyline
        points={pts}
        fill="none"
        stroke="#6366f1"
        strokeWidth={1.2}
      />
    </svg>
  );
}
