"""FastAPI application: REST + WebSocket endpoints for VLA-Shield ops."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import redis.asyncio as aioredis
from fastapi import Body, FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from shield.api.deps import get_redis, get_mysql_pool
from shield.api.evaluator import EvalInput, ShieldEvaluator


REPO_ROOT = Path(__file__).resolve().parents[3]
ONTOLOGY_DIR = REPO_ROOT / "dataset" / "ontology"


def load_rules(domain: str | None = None) -> list[dict]:
    files = [
        ONTOLOGY_DIR / "rules_physical.json",
        ONTOLOGY_DIR / "rules_semantic.json",
    ]
    merged: list[dict] = []
    for p in files:
        if p.exists():
            merged.extend(json.loads(p.read_text(encoding="utf-8")))
    if domain == "physical":
        return [r for r in merged if str(r.get("rule_id", "")).startswith("PHY.")]
    if domain == "semantic":
        return [r for r in merged if str(r.get("rule_id", "")).startswith("SEM.")]
    return merged


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> AsyncGenerator[None, None]:
    fastapi_app.state.redis = await get_redis()
    try:
        fastapi_app.state.mysql = await get_mysql_pool()
    except Exception as exc:  # pragma: no cover — optional in dev/tests
        fastapi_app.state.mysql = None
        fastapi_app.state.mysql_init_error = str(exc)
    fastapi_app.state.evaluator = ShieldEvaluator()
    try:
        yield
    finally:
        redis_client = getattr(fastapi_app.state, "redis", None)
        if redis_client is not None:
            try:
                await redis_client.aclose()
            except AttributeError:  # very old redis-py
                redis_client.close()  # type: ignore[union-attr]
        mysql_pool = getattr(fastapi_app.state, "mysql", None)
        if mysql_pool is not None:
            mysql_pool.close()
            try:
                await mysql_pool.wait_closed()
            except AttributeError:
                pass


app = FastAPI(
    title="VLA-Shield Ops API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/rules")
async def list_rules(domain: str | None = Query(default=None, pattern="^(physical|semantic)$")) -> dict:
    return {"rules": load_rules(domain)}


@app.post("/v1/evaluate")
async def evaluate_action(payload: dict = Body(default_factory=dict)) -> dict:
    robot_id = str(payload.get("robot_id", "default-robot"))
    action = [float(v) for v in payload.get("action", [])]
    if not action:
        return {"error": "empty_action", "decision": "BLOCK", "reasons": []}
    t_ns = int(payload.get("t_ns") or 0)
    try:
        sequence_id = int(payload.get("sequence_id") or 0)
    except (TypeError, ValueError):
        sequence_id = 0
    current_raw = payload.get("current_joints")
    current = [float(v) for v in current_raw] if isinstance(current_raw, list) else []
    result = app.state.evaluator.evaluate(
        EvalInput(
            robot_id=robot_id,
            action=action,
            t_ns=t_ns,
            sequence_id=sequence_id,
            current_joints=current,
        )
    )

    # Keep `/v1/robots/{id}/risk` consistent.
    r: aioredis.Redis = app.state.redis
    arbiter_key = f"arbiter:{robot_id}"
    detail_payload = json.dumps(
        {
            "ontology_ids": result["ontology_ids"],
            "latency": result["latency"],
        }
    )
    # Atomic hset + expire so dashboards never see stale arbiter data.
    pipe = r.pipeline(transaction=True)
    pipe.setex(f"risk:{robot_id}", 1, str(result["risk"]))
    pipe.hset(
        arbiter_key,
        mapping={
            "mode": result["decision"],
            "detail": detail_payload,
        },
    )
    pipe.expire(arbiter_key, 5)
    await pipe.execute()

    # Stream telemetry for the monitor WebSocket.
    telemetry = {
        "type": "telemetry",
        "robot_id": robot_id,
        "ts_ns": result["ts_ns"],
        "risk": result["risk"],
        "decision": result["decision"],
        "ontology_ids": result["ontology_ids"],
        "ontology_details": result["ontology_details"],
        "scene_rev": 0,
        "latency": result["latency"],
    }
    await r.xadd(
        f"stream:telemetry:{robot_id}",
        {"data": json.dumps(telemetry)},
        maxlen=10000,
        approximate=True,
    )
    return telemetry


@app.get("/v1/robots/{robot_id}/risk")
async def get_risk(robot_id: str) -> dict:
    """Current risk snapshot from Redis."""
    r: aioredis.Redis = app.state.redis
    score = await r.get(f"risk:{robot_id}")
    arbiter = await r.hgetall(f"arbiter:{robot_id}")
    return {
        "robot_id": robot_id,
        "risk_score": float(score) if score else None,
        "arbiter": {k.decode(): v.decode() for k, v in arbiter.items()} if arbiter else None,
    }


@app.get("/v1/robots/{robot_id}/events")
async def list_events(
    robot_id: str,
    limit: int = Query(default=100, le=1000),
    decision: str | None = Query(default=None, pattern="^(PASS|BLOCK)$"),
) -> list[dict]:
    """List recent safety events from MySQL with optional decision filter."""
    pool = app.state.mysql
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            if decision:
                await cur.execute(
                    "SELECT id, sequence_id, ts_ns, decision, risk_score, "
                    "ontology_ids, latency_total_ms, run_mode "
                    "FROM safety_events WHERE robot_id=%s AND decision=%s "
                    "ORDER BY ts_ns DESC LIMIT %s",
                    (robot_id, decision, limit),
                )
            else:
                await cur.execute(
                    "SELECT id, sequence_id, ts_ns, decision, risk_score, "
                    "ontology_ids, latency_total_ms, run_mode "
                    "FROM safety_events WHERE robot_id=%s "
                    "ORDER BY ts_ns DESC LIMIT %s",
                    (robot_id, limit),
                )
            rows = await cur.fetchall()

    return [
        {
            "event_id": row[0],
            "sequence_id": row[1],
            "ts_ns": row[2],
            "decision": row[3],
            "risk_score": row[4],
            "ontology_ids": json.loads(row[5]) if row[5] else [],
            "latency_total_ms": row[6],
            "run_mode": row[7],
        }
        for row in rows
    ]


@app.get("/v1/robots/{robot_id}/events/{event_id}")
async def get_event_detail(robot_id: str, event_id: str) -> dict:
    """Full SafetyEvent payload for a specific event."""
    pool = app.state.mysql
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT payload FROM safety_events WHERE id=%s AND robot_id=%s",
                (event_id, robot_id),
            )
            row = await cur.fetchone()
    if not row:
        return {"error": "not_found"}
    return json.loads(row[0])


@app.get("/v1/robots/{robot_id}/actions")
async def list_actions(
    robot_id: str,
    limit: int = Query(default=100, le=1000),
) -> list[dict]:
    """List recent action log entries."""
    pool = app.state.mysql
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT sequence_id, t_ns, decision, risk_score, model_id, "
                "action_dim, run_mode, latency_total_ms "
                "FROM actions_log WHERE robot_id=%s "
                "ORDER BY t_ns DESC LIMIT %s",
                (robot_id, limit),
            )
            rows = await cur.fetchall()

    return [
        {
            "sequence_id": row[0],
            "t_ns": row[1],
            "decision": row[2],
            "risk_score": row[3],
            "model_id": row[4],
            "action_dim": row[5],
            "run_mode": row[6],
            "latency_total_ms": row[7],
        }
        for row in rows
    ]


@app.websocket("/ws/telemetry/{robot_id}")
async def telemetry_ws(websocket: WebSocket, robot_id: str) -> None:
    """Stream real-time telemetry from Redis Streams via WebSocket."""
    await websocket.accept()
    r: aioredis.Redis = app.state.redis
    stream_key = f"stream:telemetry:{robot_id}"
    last_id = "$"
    try:
        while True:
            entries = await r.xread({stream_key: last_id}, count=10, block=200)
            if entries:
                for _key, messages in entries:
                    for msg_id, fields in messages:
                        last_id = msg_id
                        data = fields.get(b"data", b"{}").decode()
                        await websocket.send_text(data)
            else:
                await asyncio.sleep(0.03)
    except WebSocketDisconnect:
        pass
