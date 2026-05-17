const LABELS: Record<string, string> = {
  "PHY.COLLISION":       "Collision risk detected",
  "PHY.TIPOVER":         "Platform tip-over risk",
  "PHY.OVERLOAD":        "Actuator overload",
  "PHY.VELOCITY_LIMIT":  "Velocity cap exceeded",
  "PHY.JOINT_LIMIT":     "URDF joint limit exceeded",
  "PHY.SINGULARITY":     "Near kinematic singularity",
  "PHY.FORBIDDEN_ZONE":  "End-effector in forbidden Cartesian zone",
  "SEM.FRAGILE":         "Fragile object at risk",
  "SEM.HEAT_SOURCE":     "Heat source proximity",
  "SEM.FORBIDDEN_REGION":"Forbidden zone violation",
  "SEM.LIQUID_ELECTRICAL":"Liquid near electrical",
  "SEM.HUMAN_PROXIMITY": "Human in safety perimeter",
  "SEM.SHARP_OBJECT":    "Sharp object handling",
};

const SEVERITY_DOT: Record<string, string> = {
  "PHY.COLLISION":       "bg-red-500",
  "PHY.TIPOVER":         "bg-red-600",
  "PHY.OVERLOAD":        "bg-orange-500",
  "PHY.VELOCITY_LIMIT":  "bg-yellow-500",
  "PHY.JOINT_LIMIT":     "bg-orange-500",
  "PHY.SINGULARITY":     "bg-orange-500",
  "PHY.FORBIDDEN_ZONE":  "bg-red-600",
  "SEM.FRAGILE":         "bg-yellow-400",
  "SEM.HEAT_SOURCE":     "bg-orange-500",
  "SEM.FORBIDDEN_REGION":"bg-orange-500",
  "SEM.LIQUID_ELECTRICAL":"bg-red-600",
  "SEM.HUMAN_PROXIMITY": "bg-orange-400",
  "SEM.SHARP_OBJECT":    "bg-yellow-400",
};

const TRIGGER_LABELS: Record<string, string> = {
  "PHY.COLLISION":       "broad_phase_aabb_hit",
  "PHY.TIPOVER":         "zmp_outside_support_polygon",
  "PHY.OVERLOAD":        "joint_torque_exceeds_nominal",
  "PHY.VELOCITY_LIMIT":  "joint_velocity_exceeds_cap",
  "PHY.JOINT_LIMIT":     "joint_position_out_of_urdf_limit",
  "PHY.SINGULARITY":     "manipulability_below_threshold",
  "PHY.FORBIDDEN_ZONE":  "ee_position_inside_aabb",
  "SEM.FRAGILE":         "vfv_fragile_score_exceeds_threshold",
  "SEM.HEAT_SOURCE":     "vfv_heat_source_proximity",
  "SEM.FORBIDDEN_REGION":"semantic_zone_violation",
  "SEM.LIQUID_ELECTRICAL":"vfv_liquid_electrical_hazard",
  "SEM.HUMAN_PROXIMITY": "human_within_safety_perimeter",
  "SEM.SHARP_OBJECT":    "vfv_sharp_object_trajectory",
};

interface WhyBlockedProps {
  ontologyIds: string[];
  ontologyDetails?: Record<string, string>;
}

export default function WhyBlocked({ ontologyIds, ontologyDetails = {} }: WhyBlockedProps) {
  if (ontologyIds.length === 0) return null;

  return (
    <div className="p-4 border-b border-gray-700">
      <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Block Reasons
        <span className="ml-2 px-1.5 py-0.5 rounded bg-danger/20 text-danger text-[10px] font-bold">
          {ontologyIds.length}
        </span>
      </h2>
      <ul className="space-y-2">
        {ontologyIds.map((oid) => {
          const detail = ontologyDetails[oid];
          const dot = SEVERITY_DOT[oid] ?? "bg-danger";
          const trigger = TRIGGER_LABELS[oid];
          return (
            <li key={oid} className="flex flex-col gap-0.5">
              {/* Rule ID + label */}
              <div className="flex items-center gap-2 text-sm">
                <span className={`w-2 h-2 rounded-full ${dot} flex-shrink-0`} />
                <code className="text-gray-200 font-semibold">{oid}</code>
                <span className="text-gray-500 text-xs">{LABELS[oid] ?? "Unknown rule"}</span>
              </div>
              {/* Trigger condition */}
              {trigger && (
                <p className="pl-4 text-[10px] text-gray-500 font-mono">
                  trigger: {trigger}
                </p>
              )}
              {/* Runtime explanation (filled by arbiter) */}
              {detail && (
                <p className="pl-4 text-xs text-yellow-300/80 font-mono break-all leading-relaxed">
                  {detail}
                </p>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
