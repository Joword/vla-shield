interface HeaderProps {
  robotId: string;
  onRobotIdChange: (id: string) => void;
}

export default function Header({ robotId, onRobotIdChange }: HeaderProps) {
  return (
    <header className="flex items-center justify-between px-6 py-3 bg-panel border-b border-gray-700">
      <div className="flex items-center gap-3">
        <h1 className="text-xl font-bold tracking-tight">VLA-Shield</h1>
        <span className="text-xs bg-gray-700 px-2 py-0.5 rounded">Monitor</span>
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
