"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls, Grid } from "@react-three/drei";

export default function SceneView() {
  return (
    <Canvas camera={{ position: [2, 2, 2], fov: 50 }}>
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 5, 5]} intensity={0.8} />

      <Grid
        args={[10, 10]}
        cellSize={0.5}
        cellThickness={0.5}
        cellColor="#374151"
        sectionSize={2}
        sectionThickness={1}
        sectionColor="#4b5563"
        fadeDistance={10}
        infiniteGrid
      />

      {/* Placeholder robot base */}
      <mesh position={[0, 0.25, 0]}>
        <cylinderGeometry args={[0.1, 0.15, 0.5, 32]} />
        <meshStandardMaterial color="#3b82f6" />
      </mesh>

      {/* Placeholder end-effector */}
      <mesh position={[0.5, 0.8, 0]}>
        <sphereGeometry args={[0.06, 16, 16]} />
        <meshStandardMaterial color="#22c55e" />
      </mesh>

      {/* Placeholder obstacle */}
      <mesh position={[0.8, 0.5, 0.3]}>
        <boxGeometry args={[0.4, 1.0, 0.3]} />
        <meshStandardMaterial color="#6b7280" transparent opacity={0.6} />
      </mesh>

      <OrbitControls makeDefault />
    </Canvas>
  );
}
