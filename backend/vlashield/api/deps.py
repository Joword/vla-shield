"""Dependency factories for Redis and MySQL connections."""

from __future__ import annotations

import os

import aiomysql
import redis.asyncio as aioredis


async def get_redis() -> aioredis.Redis:
    url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
    return aioredis.from_url(url, decode_responses=False)


async def get_mysql_pool() -> aiomysql.Pool:
    return await aiomysql.create_pool(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "password"),
        db=os.getenv("MYSQL_DB", "vlashield"),
        autocommit=True,
    )
