# URDF samples

Minimal URDFs for **VLA-Shield** integration tests and documentation (no mesh assets).

- `panda_arm_simple.urdf` — six revolute joints, Franka Emika Panda–style kinematic offsets (simplified).
- `ur5_simple.urdf` — six revolute joints, Universal Robots UR5–style offsets (simplified).

These files are **not** vendor drop-ins; use official robot packages in production. They exist so `vlashield-urdf` can run FK / chain tests without external dependencies.
