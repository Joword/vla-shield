"use client";

import { Line } from "@react-three/drei";
import { useTelemetryStore, Vec3, SceneZone } from "@/store/telemetry";
import { decisionStyle } from "@/lib/decision";

/** Robot is Z-up; Three.js is Y-up. Swap y/z or the arm looks like it fell over. */
function toThree(p: Vec3): Vec3 {
  return [p[0], p[2], p[1]];
}

function ZoneBox({ zone, alert }: { zone: SceneZone; alert: boolean }) {
  const min = toThree(zone.min);
  const max = toThree(zone.max);
  const center: Vec3 = [
    (min[0] + max[0]) / 2,
    (min[1] + max[1]) / 2,
    (min[2] + max[2]) / 2,
  ];
  const size: Vec3 = [
    Math.abs(max[0] - min[0]),
    Math.abs(max[1] - min[1]),
    Math.abs(max[2] - min[2]),
  ];
  const color = alert ? "#ef4444" : "#f59e0b";
  return (
    <mesh position={center}>
      <boxGeometry args={size} />
      <meshStandardMaterial color={color} transparent opacity={alert ? 0.45 : 0.28} />
    </mesh>
  );
}

function ArmScene() {
  const skeleton = useTelemetryStore((s) => s.skeleton);
  const shadowPath = useTelemetryStore((s) => s.shadowPath);
  const zones = useTelemetryStore((s) => s.zones);
  const decision = useTelemetryStore((s) => s.decision);
  const ontologyIds = useTelemetryStore((s) => s.ontologyIds);

  const style = decisionStyle(decision);
  const chain = skeleton.map(toThree);
  const shadow = shadowPath.map(toThree);
  const ee = chain.length > 0 ? chain[chain.length - 1] : null;
  const activeZones = new Set(ontologyIds);

  return (
    <>
      {chain.length >= 2 && (
        <Line points={chain} color={style.hex} lineWidth={3} />
      )}
      {chain.map((p, i) => {
        const isEe = i === chain.length - 1;
        return (
          <mesh key={`j${i}`} position={p}>
            <sphereGeometry args={[isEe ? 0.055 : 0.032, 16, 16]} />
            <meshStandardMaterial
              color={isEe ? style.hex : "#93c5fd"}
              emissive={decision === "BLOCK" && isEe ? "#7f1d1d" : "#000000"}
            />
          </mesh>
        );
      })}
      {shadow.length >= 2 && (
        <Line
          points={shadow}
          color={style.hex}
          lineWidth={2}
          dashed
          dashSize={0.06}
          gapSize={0.04}
        />
      )}
      {ee && (
        <mesh position={ee}>
          <sphereGeometry args={[0.02, 8, 8]} />
          <meshStandardMaterial color={style.hex} />
        </mesh>
      )}
      {zones.map((zone) => (
        <ZoneBox
          key={`${zone.ontology_id}-${zone.label}`}
          zone={zone}
          alert={activeZones.has(zone.ontology_id)}
        />
      ))}
    </>
  );
}

export function SceneContent() {
  return <ArmScene />;
}
