"use client";

import { useTelemetryStore, WsStatus } from "@/store/telemetry";

interface HeaderProps {
  robotId: string;
  onRobotIdChange: (id: string) => void;
}

const STATUS_STYLE: Record<WsStatus, { label: string; className: string }> = {
  connected: {
    label: "live",
    className: "bg-green-900/50 text-green-400 border-green-700",
  },
  connecting: {
    label: "connecting",
    className: "bg-yellow-900/40 text-yellow-300 border-yellow-700 animate-pulse",
  },
  reconnecting: {
    label: "reconnecting",
    className: "bg-yellow-900/40 text-yellow-300 border-yellow-700 animate-pulse",
  },
  disconnected: {
    label: "offline",
    className: "bg-red-900/40 text-red-400 border-red-700",
  },
};

export default function Header({ robotId, onRobotIdChange }: HeaderProps) {
  const wsStatus = useTelemetryStore((s) => s.wsStatus);
  const badge = STATUS_STYLE[wsStatus];

  return (
    <header className="flex items-center justify-between px-6 py-3 bg-panel border-b border-gray-700">
      <div className="flex items-center gap-3">
        <h1 className="text-xl font-bold tracking-tight">VLA-Shield</h1>
        <span className="text-xs bg-gray-700 px-2 py-0.5 rounded">Monitor</span>
        <span
          className={`text-[10px] uppercase tracking-wider px-2 py-0.5 rounded border ${badge.className}`}
        >
          {badge.label}
        </span>
      </div>
      <div className="flex items-center gap-2">
        <label className="text-sm text-gray-400">Robot:</label>
        <input
          className="bg-surface border border-gray-600 rounded px-2 py-1 text-sm w-48"
          value={robotId}
          onChange={(e) => onRobotIdChange(e.target.value)}
        />
      </div>
    </header>
  );
}
