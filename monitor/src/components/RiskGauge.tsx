interface RiskGaugeProps {
  score: number;
  decision: string;
}

export default function RiskGauge({ score, decision }: RiskGaugeProps) {
  const color =
    decision === "BLOCK"
      ? "text-danger"
      : score > 0.5
        ? "text-warning"
        : "text-safe";

  const pct = Math.round(score * 100);

  return (
    <div className="p-4 border-b border-gray-700">
      <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Risk Score
      </h2>
      <div className="flex items-end gap-3">
        <span className={`text-5xl font-bold tabular-nums ${color}`}>
          {pct}
        </span>
        <span className="text-gray-400 text-sm mb-1">/ 100</span>
        <span
          className={`ml-auto px-3 py-1 rounded text-sm font-medium ${
            decision === "BLOCK"
              ? "bg-red-900/40 text-danger"
              : "bg-green-900/40 text-safe"
          }`}
        >
          {decision}
        </span>
      </div>
      <div className="mt-3 h-2 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-200 ${
            decision === "BLOCK"
              ? "bg-danger"
              : score > 0.5
                ? "bg-warning"
                : "bg-safe"
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
