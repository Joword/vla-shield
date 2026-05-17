# ============================================================
# VLA-Shield Unified Dockerfile
# Multi-stage build for both backend (Python) and monitor (Next.js)
# Usage:
#   docker build --target backend -f deploy/Dockerfile .
#   docker build --target monitor -f deploy/Dockerfile .
# ============================================================

# ---- Backend (Python API + ML) ----

FROM python:3.11-slim AS backend

WORKDIR /app

COPY backend/pyproject.toml ./
RUN pip install --no-cache-dir .

COPY backend/shield/ shield/

EXPOSE 8000
CMD ["uvicorn", "shield.api.app:app", "--host", "0.0.0.0", "--port", "8000"]


# ---- Monitor: dependency install ----

FROM node:20-alpine AS monitor-deps
WORKDIR /app
RUN corepack enable && corepack prepare yarn@4.0.0 --activate
COPY monitor/package.json monitor/yarn.lock* ./
RUN yarn install --immutable


# ---- Monitor: build ----

FROM node:20-alpine AS monitor-build
WORKDIR /app
RUN corepack enable && corepack prepare yarn@4.0.0 --activate
COPY --from=monitor-deps /app/node_modules ./node_modules
COPY monitor/ .
ENV NEXT_TELEMETRY_DISABLED=1
RUN yarn build


# ---- Monitor: production runner ----

FROM node:20-alpine AS monitor
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

RUN addgroup --system --gid 1001 nodejs && adduser --system --uid 1001 nextjs

COPY --from=monitor-build /app/public ./public
COPY --from=monitor-build --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=monitor-build --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000
ENV PORT=3000
CMD ["node", "server.js"]
