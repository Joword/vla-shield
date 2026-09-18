"""REST + WebSocket. Dashboard and /v1/evaluate hit this."""

from __future__ import annotations

import asyncio
import base64
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
import redis.asyncio as aioredis
from fastapi import Body, FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from shield.api.deps import get_redis, get_mysql_pool
from shield.api.evaluator import EvalInput, ShieldEvaluator


REPO_ROOT = Path(__file__).resolve().parents[3]
ONTOLOGY_DIR = REPO_ROOT / "dataset" / "ontology"


def load_rules(domain: str | None = None) -> list[dict]:
    """PHY + SEM rule dicts. domain=physical|semantic filters the prefix."""
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


def _decode_image(raw: object) -> np.ndarray | None:
    """Base64 data-URL, raw PNG/JPEG, or an HWC uint8 nested list. Else None."""
    if raw is None:
        return None
    if isinstance(raw, list):
        arr = np.asarray(raw, dtype=np.uint8)
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            return arr
        return None
    if not isinstance(raw, str) or not raw.strip():
        return None
    payload = raw.split(",", 1)[-1]
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
        return np.asarray(img, dtype=np.uint8)
    except (OSError, ValueError, TypeError):
        return None


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> AsyncGenerator[None, None]:
    """Redis always; MySQL is optional so pytest can skip the DB."""
    fastapi_app.state.redis = await get_redis()
    try:
        fastapi_app.state.mysql = await get_mysql_pool()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Dev/tests often have no MySQL. Don't take the API down with it.
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
            except AttributeError:  # old redis-py has .close(), not .aclose()
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
    version="0.6.4",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/rules")
async def list_rules(
    domain: str | None = Query(default=None, pattern="^(physical|semantic)$"),
) -> dict:
    """Ontology rules for the dashboard. Optional physical|semantic filter."""
    return {"rules": load_rules(domain)}


@app.post("/v1/evaluate")
async def evaluate_action(  # pylint: disable=too-many-locals
    payload: dict = Body(default_factory=dict),
) -> dict:
    """Run the shield on one action. Writes Redis risk + telemetry stream."""
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
    prev_raw = payload.get("prev_velocity")
    prev_velocity = [float(v) for v in prev_raw] if isinstance(prev_raw, list) else []
    hints_raw = payload.get("scene_hints") or payload.get("risk_tags") or []
    scene_hints = [str(h) for h in hints_raw] if isinstance(hints_raw, list) else []
    language_task = str(payload.get("language_task") or payload.get("task") or "")
    obstacles_raw = payload.get("obstacles")
    result = app.state.evaluator.evaluate(
        EvalInput(
            robot_id=robot_id,
            action=action,
            t_ns=t_ns,
            sequence_id=sequence_id,
            current_joints=current,
            language_task=language_task,
            scene_hints=scene_hints,
            image=_decode_image(payload.get("image")),
            obstacles=obstacles_raw if isinstance(obstacles_raw, list) else None,
            prev_velocity=prev_velocity,
        )
    )

    # So GET /v1/robots/{id}/risk sees the same decision.
    r: aioredis.Redis = app.state.redis
    arbiter_key = f"arbiter:{robot_id}"
    detail_payload = json.dumps(
        {
            "ontology_ids": result["ontology_ids"],
            "latency": result["latency"],
        }
    )
    # One pipeline: hset + expire so dashboards never see a stale arbiter hash.
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

    # Push a frame onto the Redis stream the monitor WS tails.
    telemetry = {
        "type": "telemetry",
        "robot_id": robot_id,
        "ts_ns": result["ts_ns"],
        "risk": result["risk"],
        "decision": result["decision"],
        "ontology_ids": result["ontology_ids"],
        "ontology_details": result["ontology_details"],
        "scene_rev": result.get("scene_rev", 0),
        "latency": result["latency"],
        "current_joints": result.get("current_joints", []),
        "projected_joints": result.get("projected_joints", []),
        "skeleton": result.get("skeleton", []),
        "shadow_path": result.get("shadow_path", []),
        "ee": result.get("ee"),
        "zones": result.get("zones", []),
        "vfv_backend": result.get("vfv_backend", "none"),
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
    """Latest risk + arbiter hash from Redis."""
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
    """Recent safety_events rows. Pass decision=PASS|BLOCK to filter."""
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
    """Stored JSON payload for one event."""
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
    """Recent actions_log rows for this robot."""
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
    """Tail Redis Streams over WS. last_id starts at $ so we only get new frames."""
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
