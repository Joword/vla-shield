"""Pydantic models mirroring Rust core types for Python-side validation."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RunMode(str, Enum):
    PRODUCTION = "production"
    PHYSICS_ONLY = "physics_only"
    MONITOR = "monitor"
    DISABLED = "disabled"


class OntologyNode(BaseModel):
    id: str = Field(..., pattern=r"^[A-Z]+\.[A-Z0-9_]+$")
    severity: Severity
    hard_block: bool
    title: str
    description: str
    parents: list[str] = Field(default_factory=list)


class ActionVector(BaseModel):
    t_ns: int
    sequence_id: int
    data: list[float]
    model_id: str = ""


class CollisionPair(BaseModel):
    link: str
    obstacle: str
    min_distance: float


class CollisionReport(BaseModel):
    hit: bool
    pairs: list[CollisionPair] = Field(default_factory=list)
    energy_lower_bound: float = 0.0


class SemanticRiskReport(BaseModel):
    sequence_id: int = 0
    risk_score: float = 0.0
    triggered: list[str] = Field(default_factory=list)
    stale: bool = True


class ArbiterReason(BaseModel):
    ontology_id: str
    detail: str = ""
    score: float = 0.0


class LatencyBreakdown(BaseModel):
    ingest_ms: float
    physics_ms: float
    collision_ms: float
    semantic_ms: Optional[float] = None
    total_ms: float


class SafetyEvent(BaseModel):
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
    """Single entry in the red-team dataset (JSONL format)."""

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
    """WebSocket telemetry frame sent to dashboard."""

    type: str = "telemetry"
    robot_id: str
    ts_ns: int
    risk: float
    decision: str
    ontology_ids: list[str] = Field(default_factory=list)
    scene_rev: int = 0
