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

TOTAL_STEPS     = 1_000_000
N_STEPS         = 256       # rollout length per env
BATCH_SIZE      = 256       # minibatch size for PPO updates
N_EPOCHS        = 8         # PPO epochs per rollout
GAMMA           = 0.99

# Learning rate — linearly decayed from START to START*FINAL_FRACTION
LEARNING_RATE     = 3e-4
LR_FINAL_FRACTION = 0.1

# PPO clip range — linearly decayed (large early moves, fine late tuning)
CLIP_RANGE_START = 0.2
CLIP_RANGE_FINAL = 0.1

# Entropy bonus — annealed so the agent explores strategies early
# and commits to what works late
ENT_COEF_START = 0.05
ENT_COEF_FINAL = 0.005

# Policy network — two hidden layers
NET_ARCH = [128, 128]

# Evaluation / checkpointing (in total env steps)
EVAL_FREQ       = 10_000
N_EVAL_EPISODES = 10
CHECKPOINT_FREQ = 10_000
