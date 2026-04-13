const LABELS: Record<string, string> = {
  "PHY.COLLISION": "Collision risk detected",
  "PHY.TIPOVER": "Platform tip-over risk",
  "PHY.OVERLOAD": "Actuator overload",
  "PHY.VELOCITY_LIMIT": "Velocity cap exceeded",
  "PHY.JOINT_LIMIT": "URDF joint limit exceeded",
  "PHY.SINGULARITY": "Near kinematic singularity",
  "PHY.FORBIDDEN_ZONE": "End-effector in forbidden Cartesian zone",
  "SEM.FRAGILE": "Fragile object at risk",
  "SEM.HEAT_SOURCE": "Heat source proximity",
  "SEM.FORBIDDEN_REGION": "Forbidden zone violation",
  "SEM.LIQUID_ELECTRICAL": "Liquid near electrical",
  "SEM.HUMAN_PROXIMITY": "Human in safety perimeter",
  "SEM.SHARP_OBJECT": "Sharp object handling",
};

interface WhyBlockedProps {
  ontologyIds: string[];
  ontologyDetails?: Record<string, string>;
}

function detailLine(oid: string, detail?: string): string | null {
  if (!detail) return null;
  if (oid === "PHY.JOINT_LIMIT" || oid === "PHY.SINGULARITY" || oid === "PHY.FORBIDDEN_ZONE") {
    return detail;
  }
  return null;
}

export default function WhyBlocked({ ontologyIds, ontologyDetails = {} }: WhyBlockedProps) {
  if (ontologyIds.length === 0) return null;

  return (
    <div className="p-4 border-b border-gray-700">
      <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Block Reasons
      </h2>
      <ul className="space-y-1">
        {ontologyIds.map((oid) => {
          const extra = detailLine(oid, ontologyDetails[oid]);
          return (
            <li key={oid} className="flex flex-col gap-0.5 text-sm">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-danger flex-shrink-0" />
                <code className="text-gray-300">{oid}</code>
                <span className="text-gray-500">{LABELS[oid] ?? "Unknown rule"}</span>
              </div>
              {extra && (
                <p className="pl-4 text-xs text-gray-400 font-mono break-all">{extra}</p>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
