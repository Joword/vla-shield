# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in VLA-Shield, please report it responsibly.

**Do NOT file a public GitHub issue for security vulnerabilities.**

Instead, please email: **security@vla-shield.dev**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested fix (if any)

## Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 1 week
- **Fix & disclosure**: Coordinated with reporter, typically within 30 days

## Scope

This policy covers:
- The vlashield-runtime Rust crates (`runtime/vlashield-core`, `vlashield-urdf`, `vlashield-physics`, `vlashield-collision`, `vlashield-ros2`, `vlashield-io`)
- The Python backend (`backend/`)
- The safety monitor UI (`monitor/`)
- Docker / deployment configurations (`deploy/`, `deploy/edge/`)
- Dataset tooling and URDF fixtures (insofar as they process untrusted input)
- ROS 2 message definitions (`ros2/vla_shield_msgs/`)

## Supported Versions

| Version | Supported |
|---------|-----------|
| main branch | Yes |
| Tagged releases | Latest only |
