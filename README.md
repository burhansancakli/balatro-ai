# Balatro RL — Strategy-Conditioned Hierarchical Agent

Group 7 | University of Rostock

## Architecture

```
┌──────────────────────────────────────────────────────┐
│              HIGH-LEVEL AGENT (RL / PPO)             │
│                                                      │
│  Input : observation vector (25 values)              │
│          ante, round, money, blind target,           │
│          jokers, shop, active strategy (one-hot)     │
│  Output: MultiDiscrete([10, 3])                      │
│          [0] shop action: 0=skip 1-4=buy 5-9=sell    │
│          [1] strategy:    0=FLUSH 1=PAIR 2=MULT      │
│  Frequency: once per shop visit                      │
└──────────────────────┬───────────────────────────────┘
                       │ shop action + strategy label
                       ▼
┌──────────────────────────────────────────────────────┐
│           LOW-LEVEL EXECUTOR (Calculator)            │
│                                                      │
│  Input : hand cards + strategy label + jokers        │
│  Output: best play/discard (deterministic math)      │
│  Method: brute-force combinations scored by          │
│          Chips×Mult + strategy preference,           │
│          Monte Carlo discard EV                      │
│  Frequency: every hand played                        │
└──────────────────────┬───────────────────────────────┘
                       │ card indices
                       ▼
┌──────────────────────────────────────────────────────┐
│      BACKEND (SimBackend emulator / Balatrobot)      │
│  --emulator: in-process jackdaw engine (fast)        │
│  live: N parallel Balatrobot instances               │
└──────────────────────────────────────────────────────┘
```

The agent declares a strategy at every shop decision. The declared
strategy is one-hot encoded into the observation, steers the hand
calculator until the next shop, and earns a coherence reward when the
played hand types match it — so the agent learns to align purchases
(e.g. flush jokers) with play style (flush hunting).

## Reward Structure

| Event | Reward |
|---|---|
| Each hand played (strategy coherence) | +0.0 to +0.1 |
| Blind progress shaping | +0.0 to +0.02 |
| Round survived | +0.05 |
| Selling a joker | -0.05 |
| Ante N beaten | +0.2 × (N−1), up to +2.0 for a win |
| Game over at ante 1 | -0.5 |

## File Structure

```
balatro-ai/
  strategy.py            — Strategy enum + calculator (low-level executor)
  observations.py        — Gamestate → numpy obs vector (incl. strategy one-hot)
  env.py                 — BalatroEnv Gymnasium wrapper
  train.py               — PPO training (emulator or live Balatrobot)
  episode_recorder.py    — gym.Wrapper writing episodes to JSONL per run
  replay_server.py       — standalone replay dashboard (HTTP + WebSocket)
  static/                — replay UI (index.html, app.js, style.css)
  emulator/              — in-process jackdaw engine + Balatrobot bridge
```

## Quickstart (emulator — recommended)

```bash
pip install stable-baselines3 gymnasium numpy torch websockets

# Train with the in-process emulator (4 parallel envs by default)
python train.py --emulator

# Custom length / device
python train.py --emulator -n 8 --steps 500000
python train.py --emulator --device cuda

# Monitor training
tensorboard --logdir ./logs
```

Training uses per-episode random game seeds (emulator mode), so the
agent generalizes across decks/shops instead of memorizing fixed runs.
Learning rate and clip range decay linearly; the entropy bonus anneals
from 0.05 to 0.005 so the agent explores strategies early and commits
late. Evaluation runs on unseen seeds every 10k steps and keeps the
best checkpoint in `models/best/`.

## Quickstart (live Balatrobot)

```bash
# Instances are started automatically:
python train.py -n 4 --setup-only   # create save files (once)
python train.py -n 4                # train
```

## Replay System

Every training run records completed episodes to
`logs/episodes/<timestamp>/episodes_<rank>.jsonl` (one JSON line per
episode: actions, rewards, per-hand action log with declared strategy,
final jokers).

```bash
# Browse a run in the browser
python replay_server.py --log-dir logs/episodes/<timestamp>
# → http://127.0.0.1:8765/?mode=replay
```

The dashboard shows an episode list (sort by reward / ante / steps,
filter won/lost), a step-by-step timeline with strategy badges,
cumulative hand history, and keyboard navigation (←/→).

## Baseline Comparison

The flat PPO baseline uses the same env and obs space but:
- Acts every hand (not every shop)
- Chooses card indices directly (no calculator)
- No strategy conditioning

This makes the comparison clean: same env, same seeds,
same training budget — only the architecture differs.

## Notes

Mac users: if an "Application closed unexpectedly" popup shows up
repeatedly, disable it with
`defaults write com.apple.CrashReporter DialogType none`.
