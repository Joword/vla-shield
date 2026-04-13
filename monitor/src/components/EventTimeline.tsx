"use client";

import { useTelemetryStore } from "@/store/telemetry";

interface EventTimelineProps {
  robotId: string;
}

export default function EventTimeline({ robotId: _robotId }: EventTimelineProps) {
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
        <ul className="space-y-1 text-xs font-mono">
          {recent.map((entry, i) => (
            <li key={i} className="flex justify-between text-gray-400">
              <span>{new Date(entry.ts / 1e6).toLocaleTimeString()}</span>
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
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
