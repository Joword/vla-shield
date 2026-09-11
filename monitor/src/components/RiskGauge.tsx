import { decisionStyle } from "@/lib/decision";

interface RiskGaugeProps {
  score: number;
  decision: string;
}

export default function RiskGauge({ score, decision }: RiskGaugeProps) {
  const style = decisionStyle(decision);
  const pct = Math.round(score * 100);

  return (
    <div className="p-4 border-b border-gray-700">
      <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Risk Score
      </h2>
      <div className="flex items-end gap-3">
        <span className={`text-5xl font-bold tabular-nums ${style.text}`}>
          {pct}
        </span>
        <span className="text-gray-400 text-sm mb-1">/ 100</span>
        <span className={`ml-auto px-3 py-1 rounded text-sm font-medium ${style.badge}`}>
          {decision}
        </span>
      </div>
      <div className="mt-3 h-2 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-200 ${style.bar}`}
          // PASS can be a true zero bar; anything else keeps a 4% sliver so it doesn't vanish.
          style={{ width: `${Math.max(pct, decision === "PASS" ? 0 : 4)}%` }}
        />
      </div>
    </div>
  );
}
