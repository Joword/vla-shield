"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls, Grid } from "@react-three/drei";
import { SceneContent } from "@/components/ArmScene";
import { useTelemetryStore } from "@/store/telemetry";
import { decisionStyle } from "@/lib/decision";

export default function SceneView() {
  const skeleton = useTelemetryStore((s) => s.skeleton);
  const decision = useTelemetryStore((s) => s.decision);
  const wsStatus = useTelemetryStore((s) => s.wsStatus);
  const hasArm = skeleton.length >= 2;
  const style = decisionStyle(decision);

  return (
    <div className="relative w-full h-full bg-black">
      {/* Slightly above and to the side so the table isn't edge-on. */}
      <Canvas camera={{ position: [2.2, 1.6, 2.2], fov: 50 }}>
        <color attach="background" args={["#0b1220"]} />
        <ambientLight intensity={0.45} />
        <directionalLight position={[5, 8, 5]} intensity={0.9} />
        <Grid
          args={[10, 10]}
          cellSize={0.5}
          cellThickness={0.5}
          cellColor="#1f2937"
          sectionSize={2}
          sectionThickness={1}
          sectionColor="#374151"
          fadeDistance={12}
          infiniteGrid
        />
        <SceneContent />
        <OrbitControls makeDefault />
      </Canvas>

      <div className="absolute top-3 left-3 flex flex-col gap-1 pointer-events-none">
        <span
          className={`text-[10px] uppercase tracking-wider px-2 py-0.5 rounded border w-fit ${style.badge}`}
        >
          {hasArm ? decision : "no pose"}
        </span>
        {!hasArm && (
          <span className="text-[11px] text-slate-400 bg-slate-900/70 px-2 py-1 rounded">
            Waiting for telemetry… POST /v1/evaluate to stream a skeleton.
          </span>
        )}
        {wsStatus !== "connected" && hasArm && (
          <span className="text-[11px] text-yellow-300 bg-slate-900/70 px-2 py-1 rounded">
            Stream {wsStatus} — last pose held
          </span>
        )}
      </div>
    </div>
  );
}
