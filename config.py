"""
config.py — Shared configuration constants
============================================
Used by env.py, client.py, and train.py.
"""

# ── Game defaults ───────────────────────────────────────────
DECK            = "RED"
STAKE           = "WHITE"
SEED            = "TRAIN01"

# ── Environment ─────────────────────────────────────────────
MAX_STEPS       = 500

# ── RPC / polling ───────────────────────────────────────────
RPC_RETRIES     = 3        # retry count on timeout
RPC_RETRY_WAIT  = 0.5     # seconds between retries
POLL_INTERVAL   = 0.02    # seconds between state polls
POLL_TIMEOUT    = 30.0    # max seconds to wait for a state transition

# ── Rewards ─────────────────────────────────────────────────
SURVIVAL_REWARD       = 0.05
PROGRESS_REWARD_SCALE = 0.02
CASH_OUT_SETTLE_WAIT  = 0.05
