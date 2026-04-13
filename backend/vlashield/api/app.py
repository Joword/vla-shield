"""FastAPI application: REST + WebSocket endpoints for VLA-Shield ops."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis.asyncio as aioredis
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from vlashield.api.deps import get_redis, get_mysql_pool


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.redis = await get_redis()
    app.state.mysql = await get_mysql_pool()
    yield
    app.state.redis.close()  # type: ignore[union-attr]


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
