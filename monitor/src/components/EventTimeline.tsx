"use client";

import { useTelemetryStore } from "@/store/telemetry";
import { decisionStyle } from "@/lib/decision";

export default function EventTimeline() {
  const history = useTelemetryStore((s) => s.history);
  const recent = history.slice(-20).reverse();

  return (
    <div className="p-4 flex-1 overflow-y-auto">
      <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Recent Events
      </h2>
      {recent.length === 0 ? (
        <p className="text-sm text-gray-500">Waiting for telemetry...</p>
      ) : (
        <ul className="space-y-1.5 text-xs font-mono">
          {recent.map((entry, i) => {
            const style = decisionStyle(entry.decision);
            const tags = entry.ontologyIds.slice(0, 2);
            const extra = entry.ontologyIds.length - tags.length;
            return (
              <li
                key={`${entry.ts}-${i}`}
                className="flex flex-col gap-0.5 border-b border-gray-800 pb-1 last:border-0"
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-gray-500">
                    {/* backend sends ns; Date wants ms */}
                    {new Date(entry.ts / 1e6).toLocaleTimeString()}
                  </span>
                  <span className="flex items-center gap-2">
                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${style.badge}`}>
                      {entry.decision || "—"}
                    </span>
                    <span
                      className={
                        entry.risk > 0.7
                          ? "text-danger"
                          : entry.risk > 0.4
                            ? "text-warning"
                            : "text-safe"
                      }
                    >
                      {(entry.risk * 100).toFixed(0)}%
                    </span>
                  </span>
                </div>
                {tags.length > 0 && (
                  <p className="text-[10px] text-gray-500 truncate">
                    {tags.join(" · ")}
                    {extra > 0 ? ` +${extra}` : ""}
                  </p>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
