# gunicorn.conf.py — Gunicorn configuration for Facera on Render
#
# Render free tier: 512 MB RAM, 0.1 CPU
# InsightFace buffalo_l model: ~300 MB download, ~200 MB RAM when loaded
# First inference (model warm-up): 20–60 s depending on server speed

import os

# ── Workers ──────────────────────────────────────────────────────────────────
# 1 worker on Render free tier (512 MB RAM — more workers = OOM kill)
workers = 1

# Sync worker (default) — simple, reliable; fine for 1 worker
worker_class = "sync"

# ── Timeouts ─────────────────────────────────────────────────────────────────
# Default is 30 s — way too short for first-inference model warm-up.
# 120 s gives the model time to load and run the first embedding.
timeout = 120

# Keep-alive for persistent connections
keepalive = 5

# ── Binding ──────────────────────────────────────────────────────────────────
port = int(os.environ.get("PORT", 5000))
bind = f"0.0.0.0:{port}"

# ── Logging ──────────────────────────────────────────────────────────────────
accesslog = "-"   # stdout
errorlog  = "-"   # stderr
loglevel  = "info"
