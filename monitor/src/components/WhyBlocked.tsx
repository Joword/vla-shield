const LABELS: Record<string, string> = {
  "PHY.COLLISION": "Collision risk detected",
  "PHY.TIPOVER": "Platform tip-over risk",
  "PHY.OVERLOAD": "Actuator overload",
  "PHY.VELOCITY_LIMIT": "Velocity cap exceeded",
  "SEM.FRAGILE": "Fragile object at risk",
  "SEM.HEAT_SOURCE": "Heat source proximity",
  "SEM.FORBIDDEN_REGION": "Forbidden zone violation",
  "SEM.LIQUID_ELECTRICAL": "Liquid near electrical",
  "SEM.HUMAN_PROXIMITY": "Human in safety perimeter",
  "SEM.SHARP_OBJECT": "Sharp object handling",
};

interface WhyBlockedProps {
  ontologyIds: string[];
}

export default function WhyBlocked({ ontologyIds }: WhyBlockedProps) {
  if (ontologyIds.length === 0) return null;

  return (
    <div className="p-4 border-b border-gray-700">
      <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Block Reasons
      </h2>
      <ul className="space-y-1">
        {ontologyIds.map((oid) => (
          <li key={oid} className="flex items-center gap-2 text-sm">
            <span className="w-2 h-2 rounded-full bg-danger flex-shrink-0" />
            <code className="text-gray-300">{oid}</code>
            <span className="text-gray-500">
              {LABELS[oid] ?? "Unknown rule"}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
