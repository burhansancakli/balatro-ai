"""
config.py — Shared configuration constants
============================================
Used by env.py and train.py.
"""

from pathlib import Path

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

# ── Training ────────────────────────────────────────────────
SAVE_DIR        = Path.cwd() / "balatro_saves"
MODEL_DIR       = "./models"
LOG_DIR         = "./logs"
TOTAL_STEPS     = 100_000
N_STEPS         = 256
BATCH_SIZE      = 64
N_EPOCHS        = 10
LEARNING_RATE   = 0.0003
GAMMA           = 0.99
EVAL_FREQ       = 256
CHECKPOINT_FREQ = 2_000
