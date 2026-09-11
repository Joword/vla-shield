"""Redis / MySQL factories. Env vars, then the usual localhost defaults."""

from __future__ import annotations

import os

import aiomysql
import redis.asyncio as aioredis


async def get_redis() -> aioredis.Redis:
    """REDIS_URL, or redis://127.0.0.1:6379."""
    url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
    return aioredis.from_url(url, decode_responses=False)


async def get_mysql_pool() -> aiomysql.Pool:
    """MYSQL_* env vars, then the usual localhost root/password/shield."""
    return await aiomysql.create_pool(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "password"),
        db=os.getenv("MYSQL_DB", "shield"),
        autocommit=True,
    )
