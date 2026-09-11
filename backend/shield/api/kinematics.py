"""FK for the monitor and the Python evaluator.

URDF chain first (same as shield-urdf: Z-up, metres). Yaw/pitch sketch
only if there's no URDF. Monitor flips to Three.js Y-up at render time.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# UR5-ish link lengths (base lift is applied separately). Extra DoF
# past 6 get 8 cm distal links.
_BASE_LINKS = (0.12, 0.35, 0.30, 0.12, 0.10, 0.08)
_BASE_LIFT = 0.12

# Table-side demo box — same placeholder SceneView used to hard-code.
DEFAULT_ZONES: list[dict] = [
    {
        "min": [0.55, -0.15, 0.0],
        "max": [0.95, 0.25, 0.90],
        "label": "obstacle",
        "ontology_id": "PHY.FORBIDDEN_ZONE",
    }
]


def link_lengths_for_dof(dof: int) -> list[float]:
    """UR5-ish sketch lengths. Extra DoF past 6 get 8 cm distal links."""
    if dof <= 0:
        return []
    if dof <= len(_BASE_LINKS):
        return list(_BASE_LINKS[:dof])
    return list(_BASE_LINKS) + [0.08] * (dof - len(_BASE_LINKS))


def fk_skeleton(joints: list[float]) -> list[list[float]]:
    """Sketch waypoints [origin, base, j1, …, jN], robot XYZ, Z-up."""
    dof = len(joints)
    if dof == 0:
        return [[0.0, 0.0, 0.0], [0.0, 0.0, _BASE_LIFT]]
    lengths = link_lengths_for_dof(dof)
    points: list[list[float]] = [[0.0, 0.0, 0.0], [0.0, 0.0, _BASE_LIFT]]
    x, y, z = 0.0, 0.0, _BASE_LIFT
    yaw = 0.0
    pitch = 0.0
    for i, (q, length) in enumerate(zip(joints, lengths)):
        if i == 0:
            yaw += q
        elif i == 1:
            pitch += q
        elif i % 2 == 0:
            yaw += 0.5 * q
        else:
            pitch += 0.5 * q
        cos_p = math.cos(pitch)
        x += length * cos_p * math.cos(yaw)
        y += length * cos_p * math.sin(yaw)
        z += length * math.sin(pitch)
        z = max(z, 0.02)
        points.append([x, y, z])
    return points


def ee_of(joints: list[float]) -> list[float]:
    """EE of the yaw/pitch sketch — last point of fk_skeleton."""
    return fk_skeleton(joints)[-1]


def shadow_polyline(joint_samples: list[list[float]]) -> list[list[float]]:
    """EE XYZ per shadow sample. Empty samples are skipped."""
    return [ee_of(sample) for sample in joint_samples if sample]


def zones_for(ontology_ids: list[str]) -> list[dict]:
    """Demo table box, plus extra AABBs when SEM.* ids fire."""
    zones = [dict(z) for z in DEFAULT_ZONES]
    triggered = set(ontology_ids)
    if "SEM.HEAT_SOURCE" in triggered:
        zones.append(
            {
                "min": [0.20, 0.30, 0.20],
                "max": [0.50, 0.60, 0.70],
                "label": "heat_source",
                "ontology_id": "SEM.HEAT_SOURCE",
            }
        )
    if "SEM.FORBIDDEN_REGION" in triggered:
        zones.append(
            {
                "min": [-0.20, 0.40, 0.00],
                "max": [0.20, 0.80, 0.50],
                "label": "forbidden_region",
                "ontology_id": "SEM.FORBIDDEN_REGION",
            }
        )
    if "SEM.HUMAN_PROXIMITY" in triggered:
        zones.append(
            {
                "min": [-0.35, -0.55, 0.00],
                "max": [0.35, -0.15, 1.20],
                "label": "human_perimeter",
                "ontology_id": "SEM.HUMAN_PROXIMITY",
            }
        )
    return zones


REPO_ROOT = Path(__file__).resolve().parents[3]
UR5_URDF = REPO_ROOT / "dataset" / "urdf" / "ur5_simple.urdf"
PANDA_URDF = REPO_ROOT / "dataset" / "urdf" / "panda_arm_simple.urdf"
_SYNTH_RADIUS = 0.045


def default_urdf_for_dof(dof: int) -> tuple[Path, str, str] | None:
    """Bundled UR5 (6) or Panda (7). Path + root + EE link names, or None."""
    if dof == 6 and UR5_URDF.is_file():
        return UR5_URDF, "base_link", "wrist_3_link"
    if dof == 7 and PANDA_URDF.is_file():
        return PANDA_URDF, "panda_link0", "panda_hand"
    return None


def _vec3(text: str | None, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> list[float]:
    if not text:
        return list(default)
    parts = text.split()
    if len(parts) != 3:
        return list(default)
    return [float(p) for p in parts]


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = sum(a[i][k] * b[k][j] for k in range(4))
    return out


def _eye() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _xyz_rpy_matrix(xyz: list[float], rpy: list[float]) -> list[list[float]]:  # pylint: disable=too-many-locals
    cr, cp, cy = math.cos(rpy[0]), math.cos(rpy[1]), math.cos(rpy[2])
    sr, sp, sy = math.sin(rpy[0]), math.sin(rpy[1]), math.sin(rpy[2])
    # R = Rz(yaw) * Ry(pitch) * Rx(roll) — same as nalgebra from_euler_angles(r,p,y)
    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr
    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr
    return [
        [r00, r01, r02, xyz[0]],
        [r10, r11, r12, xyz[1]],
        [r20, r21, r22, xyz[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _axis_angle(axis: list[float], q: float) -> list[list[float]]:
    n = math.sqrt(sum(c * c for c in axis)) or 1.0
    x, y, z = axis[0] / n, axis[1] / n, axis[2] / n
    c, s = math.cos(q), math.sin(q)
    t = 1.0 - c
    return [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0.0],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0.0],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


@dataclass(frozen=True)
class _Joint:
    name: str
    parent: str
    child: str
    origin_xyz: list[float]
    origin_rpy: list[float]
    axis: list[float]


@dataclass
class UrdfChain:
    """Serial revolute chain from a URDF. 4×4 frames, Z-up, metres."""

    joints: list[_Joint]
    root: str
    ee: str

    @property
    def dof(self) -> int:
        """How many revolute/continuous joints we actually drive."""
        return len(self.joints)

    def frames(self, q: list[float]) -> list[tuple[str, list[list[float]]]]:
        """(link, 4×4) from root through each child at q."""
        world = _eye()
        frames = [(self.root, world)]
        n = min(len(q), len(self.joints))
        for i in range(n):
            j = self.joints[i]
            origin = _xyz_rpy_matrix(j.origin_xyz, j.origin_rpy)
            motion = _axis_angle(j.axis, float(q[i]))
            world = _matmul(world, _matmul(origin, motion))
            frames.append((j.child, world))
        return frames

    def skeleton(self, q: list[float]) -> list[list[float]]:
        """Link origins, root first."""
        pts = []
        for _, m in self.frames(q):
            pts.append([m[0][3], m[1][3], m[2][3]])
        return pts

    def link_aabbs(  # pylint: disable=too-many-locals
        self, q: list[float], radius: float = _SYNTH_RADIUS
    ) -> list[tuple[str, list[float], list[float]]]:
        """Fat world AABBs per link at q — origin-to-next plus radius."""
        frames = self.frames(q)
        boxes: list[tuple[str, list[float], list[float]]] = []
        for idx, (name, m) in enumerate(frames):
            if idx + 1 < len(self.joints):
                span = self.joints[idx].origin_xyz if idx == 0 else self.joints[idx].origin_xyz
            else:
                span = [0.06, 0.0, 0.0]
            # origin + next (or span) plus radius. Fat on purpose.
            ox, oy, oz = m[0][3], m[1][3], m[2][3]
            if idx + 1 < len(frames):
                nxt = frames[idx + 1][1]
                nx, ny, nz = nxt[0][3], nxt[1][3], nxt[2][3]
            else:
                nx, ny, nz = ox + span[0], oy + span[1], oz + span[2]
            mn = [min(ox, nx) - radius, min(oy, ny) - radius, min(oz, nz) - radius]
            mx = [max(ox, nx) + radius, max(oy, ny) + radius, max(oz, nz) + radius]
            boxes.append((name, mn, mx))
        return boxes


def _chain_between(joints: dict[str, _Joint], root: str, ee: str) -> list[_Joint]:
    by_child = {j.child: j for j in joints.values()}
    ordered_rev: list[_Joint] = []
    cur = ee
    while cur != root:
        j = by_child.get(cur)
        if j is None:
            raise ValueError(f"no joint whose child is {cur}")
        ordered_rev.append(j)
        cur = j.parent
    ordered_rev.reverse()
    return ordered_rev


@lru_cache(maxsize=8)
def load_urdf_chain(path: str, root: str, ee: str) -> UrdfChain:
    """Revolute/continuous joints from root → ee. Skips fixed/prismatic."""
    tree = ET.parse(path)
    robot = tree.getroot()
    joints: dict[str, _Joint] = {}
    for node in robot.findall("joint"):
        jtype = node.attrib.get("type", "")
        if jtype not in {"revolute", "continuous"}:
            continue
        parent_el = node.find("parent")
        child_el = node.find("child")
        origin = node.find("origin")
        axis = node.find("axis")
        parent = "" if parent_el is None else parent_el.attrib.get("link", "")
        child = "" if child_el is None else child_el.attrib.get("link", "")
        joints[node.attrib.get("name", "")] = _Joint(
            name=node.attrib.get("name", ""),
            parent=parent,
            child=child,
            origin_xyz=_vec3(None if origin is None else origin.attrib.get("xyz")),
            origin_rpy=_vec3(None if origin is None else origin.attrib.get("rpy")),
            axis=_vec3(None if axis is None else axis.attrib.get("xyz"), (0.0, 0.0, 1.0)),
        )
    chain = _chain_between(joints, root, ee)
    return UrdfChain(joints=chain, root=root, ee=ee)


def aabb_intersects(
    a_min: list[float],
    a_max: list[float],
    b_min: list[float],
    b_max: list[float],
) -> bool:
    """Inclusive AABB overlap."""
    return (
        a_min[0] <= b_max[0]
        and a_max[0] >= b_min[0]
        and a_min[1] <= b_max[1]
        and a_max[1] >= b_min[1]
        and a_min[2] <= b_max[2]
        and a_max[2] >= b_min[2]
    )


def point_in_aabb(p: list[float], bmin: list[float], bmax: list[float]) -> bool:
    """Inclusive point-in-box."""
    return (
        bmin[0] <= p[0] <= bmax[0]
        and bmin[1] <= p[1] <= bmax[1]
        and bmin[2] <= p[2] <= bmax[2]
    )


def collision_pairs(
    joints: list[float],
    obstacles: list[dict],
    chain: UrdfChain | None = None,
) -> list[tuple[str, str]]:
    """(link, obstacle) AABB hits.

    Skip PHY.FORBIDDEN_ZONE here — that's an EE point check in forbidden_zone_hits.
    """
    scene = [o for o in obstacles if o.get("ontology_id") != "PHY.FORBIDDEN_ZONE"]
    if chain is None:
        pts = fk_skeleton(joints)
        ee = pts[-1] if pts else [0.0, 0.0, 0.0]
        r = 0.06
        boxes = [("ee", [ee[0] - r, ee[1] - r, ee[2] - r], [ee[0] + r, ee[1] + r, ee[2] + r])]
    else:
        boxes = chain.link_aabbs(joints)
    hits: list[tuple[str, str]] = []
    for link, mn, mx in boxes:
        for obs in scene:
            omin = [float(v) for v in obs["min"]]
            omax = [float(v) for v in obs["max"]]
            if aabb_intersects(mn, mx, omin, omax):
                hits.append((link, str(obs.get("label") or obs.get("id") or "obstacle")))
    return hits


def forbidden_zone_hits(
    joints: list[float],
    obstacles: list[dict],
    chain: UrdfChain | None = None,
) -> list[str]:
    """Labels of PHY.FORBIDDEN_ZONE boxes that contain the EE."""
    zones = [o for o in obstacles if o.get("ontology_id") == "PHY.FORBIDDEN_ZONE"]
    if not zones:
        return []
    if chain is not None:
        pts = chain.skeleton(joints)
    else:
        pts = fk_skeleton(joints)
    ee = pts[-1] if pts else [0.0, 0.0, 0.0]
    hits: list[str] = []
    for z in zones:
        if point_in_aabb(ee, [float(v) for v in z["min"]], [float(v) for v in z["max"]]):
            hits.append(str(z.get("label") or z.get("id") or "forbidden"))
    return hits


def parse_obstacles(raw: object) -> list[dict]:
    """Evaluate-payload obstacle dicts. Empty / junk → no collision scene."""
    if not isinstance(raw, list) or not raw:
        return []
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        mn = item.get("min")
        mx = item.get("max")
        if not (isinstance(mn, list) and isinstance(mx, list) and len(mn) == 3 and len(mx) == 3):
            continue
        out.append(
            {
                "min": [float(v) for v in mn],
                "max": [float(v) for v in mx],
                "label": str(item.get("label") or item.get("id") or "obstacle"),
                "ontology_id": str(item.get("ontology_id") or "PHY.COLLISION"),
            }
        )
    return out


def obstacles_as_tuples(
    obstacles: list[dict],
) -> list[tuple[str, float, float, float, float, float, float, str]]:
    """Rows for ShieldPipeline.set_obstacles.

    Last field is the ontology id so Rust can split like we do: PHY.FORBIDDEN_ZONE
    is an EE point check, everything else is a collision body.
    """
    rows = []
    for z in obstacles:
        mn, mx = z["min"], z["max"]
        rows.append(
            (
                str(z.get("label") or "obstacle"),
                float(mn[0]),
                float(mn[1]),
                float(mn[2]),
                float(mx[0]),
                float(mx[1]),
                float(mx[2]),
                str(z.get("ontology_id") or "PHY.COLLISION"),
            )
        )
    return rows


def fk_skeleton_for(joints: list[float], chain: UrdfChain | None = None) -> list[list[float]]:
    """URDF skeleton when the chain matches, else the yaw/pitch sketch."""
    if chain is not None and chain.dof == len(joints):
        return chain.skeleton(joints)
    spec = default_urdf_for_dof(len(joints))
    if spec is not None:
        path, root, ee = spec
        try:
            return load_urdf_chain(str(path), root, ee).skeleton(joints)
        except (OSError, ValueError, ET.ParseError):
            pass
    return fk_skeleton(joints)
