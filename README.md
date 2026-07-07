# Balatro RL

A reinforcement-learning training stack for Balatro that combines a Gymnasium environment, a deterministic hand executor, and PPO-based training. The project supports both live Balatrobot instances and a built-in emulator backend for fast local experimentation.

## What the project does

- Wraps Balatro as a Gymnasium environment with a discrete action space.
- Uses a deterministic calculator for hand selection so the RL policy can focus on shop decisions.
- Supports training with Stable-Baselines3 PPO in parallel environments.
- Includes optional emulator mode for headless runs without launching Balatrobot.
- Provides callbacks for game status tracking, shop action logging, and checkpointing.

## Current architecture

1. The environment loads a fresh save and polls the game state from the backend.
2. The RL agent selects an action such as skipping, buying a shop card, or selling a joker.
3. The environment executes the chosen action and then plays the remainder of the ante with the deterministic hand calculator.
4. Training runs through vectorized environments and saves checkpoints, evaluation logs, and TensorBoard output.

## Repository structure

```text
.
├── config.py                # Shared training and environment settings
├── env.py                   # Gymnasium environment wrapper
├── instance_manager.py      # Balatrobot process/port management
├── observations.py          # Observation vector construction
├── strategy.py              # Deterministic hand-planning logic
├── train.py                 # PPO training entrypoint
├── emulator/                # In-process emulator backend
├── tests/                   # Unit tests for strategy, rewards, and observations
├── balatro_saves/           # Seeded save files used by training
├── models/                  # Saved PPO models and checkpoints
├── logs/                    # TensorBoard and training logs
└── runs/                    # Run metadata and experiment output
```

## Requirements

- Python 3.12+
- The project is configured via [pyproject.toml](pyproject.toml)

Install dependencies with:

```bash
pip install -e .
```

## Quick start

### 1. Training examples

```bash
python train.py --headless
python train.py --instances 4 --emulator
python train.py --instances 4 --emulator --resume ./models/PPO_77_4000_steps.zip
```

### 2. Emulator mode (no Balatrobot required)

```bash
emulator run --mode play
emulator run --mode play --host 127.0.0.1 --port 12346
emulator run --mode watch
emulator run --mode watch --agent smart
```

### 3. Live Balatrobot mode

```bash
python train.py
```

### 4. Custom instance count or seeds

```bash
python train.py -n 4
python train.py --seeds TRAIN01 TRAIN02 TRAIN03
```

## Useful training options

- `--emulator`: use the built-in emulator backend
- `--resume path/to/model.zip`: continue training from a saved checkpoint
- `--delay 0.5`: pause briefly between environment steps for visualization
- `-n 4`: run multiple parallel environments

## Monitoring

TensorBoard logs are written under the `logs/` directory:

```bash
tensorboard --logdir ./logs
```

## Testing

Run the unit test suite with:

```bash
pytest
```

To exclude benchmark and live-only tests:

```bash
pytest -m "not benchmark and not live"
```

## Notes

- The training configuration is centralized in [config.py](config.py).
- The project is designed for research and experimentation; expect to adjust hyperparameters and environment settings for your hardware and target behavior.
- On macOS, Balatrobot may occasionally trigger crash-report dialogs; these are external to the RL stack itself.
