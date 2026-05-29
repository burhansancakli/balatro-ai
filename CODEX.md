# Codex Runbook

This project trains a strategy-conditioned hierarchical PPO agent for Balatro.
Use this file as the first stop before editing or testing.

## Environment

- Windows only.
- Python 3.11 only.
- Use `.\venv\Scripts\python.exe`.
- Balatrobot version: `1.4.1`.
- Steamodded version: `1.0.0-beta-1224a`.
- Do not use Balatrobot headless mode on Windows.

## Architecture

- High-level agent: PPO chooses one strategy per ante.
- Action space: `0=FLUSH_BUILD`, `1=PAIR_BUILD`, `2=MULT_BUILD`.
- Low-level executor: deterministic calculator in `strategy.py`.
- Low-level card play must not use RL.
- Reset must use Balatrobot `load()`, not `start()`.

## Safe Commands

Run fast offline tests:

```powershell
.\scripts\test.bat
```

Diagnose local Python setup:

```powershell
.\scripts\doctor.bat
```

Open TensorBoard:

```powershell
.\scripts\tensorboard.bat
```

Run live Balatrobot smoke tests only after a server is running:

```powershell
.\scripts\smoke.bat
```

Run one full live Gym environment step:

```powershell
.\scripts\env_check.bat
```

Start one fast Balatrobot instance:

```powershell
.\scripts\start_fast_1.bat
```

All project launchers pin Balatrobot to `1.4.1`.

If normal rendering is slow, try render-on-api mode:

```powershell
.\scripts\start_api_fast_1.bat
```

Time a cheap RPC call:

```powershell
.\scripts\rpc_timing.bat gamestate
```

## Never Change Casually

- Do not add more than 3 strategies.
- Do not remove card suits.
- Do not replace `load()` reset with `start()`.
- Do not add an LLM agent level.
- Do not use `--headless`.
- Do not set `unlocked=false` in the Lua mod.
- Do not call `set_ability` in Lua.
- Do not return `nil` from shop-card creation hooks.

## Testing Policy

- Put pure Python tests in `tests/`; these must not require Balatrobot.
- Put RPC/Balatrobot tests in `tests_live/`; these may require a running server.
- After changing `env.py`, `strategy.py`, `observations.py`, or `jokers.py`, run `.\scripts\test.bat`.
- After changing RPC flow, also run `.\scripts\smoke.bat` with Balatrobot running.
- Before training, run `.\scripts\env_check.bat` with Balatrobot running.

## Research Notes

- Keep fixed seeds `TRAIN01` through `TRAIN04` for reproducibility.
- Keep hierarchical PPO and flat PPO baseline comparable by using the same seeds and training budget.
- Log paper metrics through TensorBoard under `research/*`.
