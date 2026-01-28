# Rate Limiting

## Overview
- Requests are limited to 120 requests per minute per API key.
- Burst handling allows up to 30 additional requests in a 60 second window.

## Implementation
- The rate limiter is implemented in `backend/src/multi_rag/api/middleware/rate_limit.py`.
- It uses an in-memory token bucket in development.
- In production, it uses Redis with key prefix `ratelimit:`.

## Configuration
- `RATE_LIMIT_PER_MINUTE` sets the steady-state limit.
- `RATE_LIMIT_BURST` sets the burst allowance.
