"""Pydantic twins of the Rust types. Validate JSON on the Python side with these."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic.fields import Field
from pydantic.main import BaseModel


class Severity(str, Enum):
    """info → critical. Same labels as dataset/ontology."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RunMode(str, Enum):
    """Which checks the hot path actually runs."""

    PRODUCTION = "production"
    PHYSICS_ONLY = "physics_only"
    MONITOR = "monitor"
    DISABLED = "disabled"


class OntologyNode(BaseModel):
    """One PHY.* / SEM.* node from physical.json / semantic.json."""

    id: str = Field(..., pattern=r"^[A-Z]+\.[A-Z0-9_]+$")
    severity: Severity
    hard_block: bool
    title: str
    description: str
    parents: list[str] = Field(default_factory=list)


class RuleAction(str, Enum):
    """What the arbiter does when a rule fires."""

    BLOCK = "block"
    CLAMP = "clamp"
    WARN = "warn"


class RuleEntry(BaseModel):
    """One rule from dataset/ontology/rules_*.json."""

    rule_id: str = Field(..., pattern=r"^[A-Z]+\.[A-Z0-9_]+$")
    trigger_condition: str = Field(..., min_length=1)
    threshold: dict[str, Any] = Field(...)
    action: RuleAction
    severity: Severity
    hard_block: bool = False
    explanation_template: str = Field(..., min_length=1)
    applies_to: list[str] = Field(default_factory=list)
    disabled: bool = False


class ActionVector(BaseModel):
    """One VLA command: timestamp, seq, joint velocities."""

    t_ns: int
    sequence_id: int
    data: list[float]
    model_id: str = ""


class CollisionPair(BaseModel):
    """One link vs one obstacle from the AABB sweep."""

    link: str
    obstacle: str
    min_distance: float


class CollisionReport(BaseModel):
    """Broad-phase result. Empty pairs = nothing overlapped."""

    hit: bool
    pairs: list[CollisionPair] = Field(default_factory=list)
    energy_lower_bound: float = 0.0


class SemanticRiskReport(BaseModel):
    """VFV / hint prior. stale=True until an async pass lands."""

    sequence_id: int = 0
    risk_score: float = 0.0
    triggered: list[str] = Field(default_factory=list)
    stale: bool = True


class ArbiterReason(BaseModel):
    """One ontology hit the arbiter ranked."""

    ontology_id: str
    detail: str = ""
    score: float = 0.0


class LatencyBreakdown(BaseModel):
    """Per-stage ms. None means that stage didn't run."""

    ingest_ms: float = 0.0
    urdf_fk_ms: Optional[float] = None
    physics_ms: float = 0.0
    collision_ms: float = 0.0
    tf2_ms: Optional[float] = None
    arbiter_ms: float = 0.0
    shadow_ms: Optional[float] = None
    total_ms: float = 0.0


class SafetyEvent(BaseModel):
    """One audit row: decision + reasons + latency."""

    event_id: str
    ts_ns: int
    robot_id: str
    sequence_id: int
    decision: str
    action_hash: str
    mode: RunMode
    reasons: list[ArbiterReason] = Field(default_factory=list)
    latency: Optional[LatencyBreakdown] = None


class RedTeamEntry(BaseModel):
    """One red-team JSONL row."""

    id: str = Field(..., pattern=r"^[a-z]{2}-[a-z0-9]{6}$")
    split: str = Field(default="train", pattern=r"^(train|val|test)$")
    locale: str = "en"
    instruction: str = Field(..., min_length=1)
    task_context: str = ""
    risk_tags: list[str] = Field(default_factory=list)
    expected_outcome: str = Field(
        default="refuse_or_safe_alternative",
        pattern=r"^(refuse_or_safe_alternative|proceed_with_caution|proceed_safe)$",
    )
    action_gold: Optional[list[float]] = None
    annotator: str = ""
    source: str = "manual"
    version: str = "v0.1"


class TelemetryMessage(BaseModel):
    """One WS frame the dashboard listens for."""

    type: str = "telemetry"
    robot_id: str
    ts_ns: int
    risk: float
    decision: str
    ontology_ids: list[str] = Field(default_factory=list)
    scene_rev: int = 0
