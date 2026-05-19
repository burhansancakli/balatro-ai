# Balatro RL — Strategy-Conditioned Hierarchical Agent

Group 7 | University of Rostock

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              HIGH-LEVEL AGENT (RL / PPO)             │
│                                                      │
│  Input : observation vector (35 values)             │
│  Output: strategy label  Discrete(3)                │
│          0=FLUSH_BUILD 1=PAIR_BUILD 2=MULT_BUILD    │
│  Frequency: once per ante (8 decisions per run)     │
└──────────────────────┬──────────────────────────────┘
                       │ strategy label
                       ▼
┌─────────────────────────────────────────────────────┐
│           LOW-LEVEL EXECUTOR (Calculator)            │
│                                                      │
│  Input : hand cards + strategy label               │
│  Output: best 5-card play (deterministic math)     │
│  Method: brute-force C(8,5)=56 combinations        │
│          scored by Chips×Mult + strategy preference │
│  Frequency: every hand played                       │
└──────────────────────┬──────────────────────────────┘
                       │ card indices
                       ▼
┌─────────────────────────────────────────────────────┐
│              BALATROBOT (Real Game)                  │
│  4 parallel instances, ports 12346-12349            │
│  Fixed seeds for reproducibility                    │
│  Fast mode: --fast --no-shaders --fps-cap 1000      │
└─────────────────────────────────────────────────────┘
```

## Reward Structure

| Event | Reward |
|---|---|
| Each hand played (coherence) | +0.0 to +0.1 |
| Ante 2 beaten | +0.2 |
| Ante 3 beaten | +0.4 |
| Ante 4 beaten | +0.6 |
| Ante 5 beaten | +0.8 |
| Ante 6 beaten | +1.0 |
| Ante 7 beaten | +1.5 |
| Ante 8 beaten (win) | +2.0 |
| Game over at ante 1 | -0.5 |

## File Structure

```
balatro_rl/
  strategy.py      — Strategy enum + calculator (low-level executor)
  observations.py  — Gamestate → numpy obs vector
  env.py           — BalatroEnv Gymnasium wrapper
  train.py         — PPO training with 4 parallel instances
```

## Quickstart

```bash
# 1. Install dependencies
pip install stable-baselines3 gymnasium requests numpy

# 2. Start 4 Balatrobot instances (separate terminals)

uvx balatrobot serve --fast --no-shaders --fps-cap 1000 --gamespeed 4 --port 12346
uvx balatrobot serve --fast --no-shaders --fps-cap 1000 --gamespeed 4 --port 12347
uvx balatrobot serve --fast --no-shaders --fps-cap 1000 --gamespeed 4 --port 12348
uvx balatrobot serve --fast --no-shaders --fps-cap 1000 --gamespeed 4 --port 12349


# HEADLESS

uvx balatrobot serve --headless --fast --no-shaders --fps-cap 1000 --gamespeed 4 --port 12347
uvx balatrobot serve --headless --fast --no-shaders --fps-cap 1000 --gamespeed 4 --port 12348
uvx balatrobot serve --headless --fast --no-shaders --fps-cap 1000 --gamespeed 4 --port 12349
uvx balatrobot serve --headless --fast --no-shaders --fps-cap 1000 --gamespeed 4 --port 12346

# 3. Create save files (once)
python train.py --setup-only

# 4. Train
python train.py

# 5. Monitor
tensorboard --logdir ./logs
```

# Notes

If you're a Mac user, you a popup about "Application closed unexpectedly" could repeteadly show up. In order to disable this, you can use the following terminal code „defaults write com.apple.CrashReporter DialogType none”


## Baseline Comparison

The flat PPO baseline uses the same env and obs space but:
- Acts every hand (not every ante)
- Chooses card indices directly (no calculator)
- No strategy conditioning

This makes the comparison clean: same env, same seeds,
same training budget — only the architecture differs.


Oops! The game crashed:
card.lua:276: attempt to index local 'center' (a nil value)

Additional Context:
Balatro Version: 1.0.1o-FULL
Modded Version: 1.0.0~BETA-1620a-STEAMODDED
LÖVE Version: 11.5.0
Lovely Version: 0.9.0
Platform: Windows
Steamodded Mods:
    1: BalatroBot by S1M0N38, stirby, phughesion, besteon, giewev [ID: balatrobot, Version: 1.4.0]
    2: DebugPlus by WilsontheWolf [ID: DebugPlus, Version: 1.5.2, Uses Lovely]
    3: Balatro RL Simplifier by Group7 [ID: BalatroRL, Priority: -10, Version: 1.0.0]
Lovely Mods:

Stack Traceback
===============
(3) LÖVE metamethod at file 'boot.lua:352'
Local variables:
 errhand = Lua function '(LÖVE Function)' (defined at line 616 of chunk [lovely debugplus.console "debugplus/console.lua"])
 handler = Lua function '(LÖVE Function)' (defined at line 616 of chunk [lovely debugplus.console "debugplus/console.lua"])
(4) Lua upvalue 'set_ability' at file 'card.lua:276'
Local variables:
 self = table: 0x392829d0  {click_offset:table: 0x3941b4d0, children:table: 0x3946d388, ambient_tilt:0.2, T:table: 0x3902b828, offset:table: 0x38deefe0, sell_cost:2, role:table: 0x3901c7d8 (more...)}
 center = nil
 initial = nil
 delay_sprites = nil
 X = number: 12.5287
 Y = number: 4.15195
 W = number: 2.04878
 H = number: 2.75122
 old_center = table: 0x391eb1f8  {alerted:true, _saved_d_u:true, unlocked:true, _u:true, effect:Hand Size Mult, cost:5, order:16, _d:false, blueprint_compat:true, eternal_compat:true (more...)}
 was_added_to_deck = boolean: false
 (*temporary) = boolean: false
 (*temporary) = table: 0x391e5980  {j_mime:table: 0x391eb938, j_ring_master:table: 0x391fc0a0, b_challenge:table: 0x39210000, c_medium:table: 0x392081d8, p_standard_jumbo_2:table: 0x392153e8 (more...)}
 (*temporary) = number: nan
 (*temporary) = string: "b_zodiac"
 (*temporary) = table: 0x3920f598  {alerted:true, _saved_d_u:true, original_key:zodiac, inject:function: 0x392905f8, unlocked:true, unlock_condition:table: 0x3920f798, _discovered_unlocked_overwritten:true (more...)}
 (*temporary) = C function: builtin#6
 (*temporary) = table: 0x39014830  {1:table: 0x3983eba0, 2:table: 0x398d5a88, 3:table: 0x3902e278}
 (*temporary) = number: 3
 (*temporary) = nil
 (*temporary) = string: "attempt to index local 'center' (a nil value)"
(5) Lua method 'set_ability' at Steamodded file 'src/overrides.lua:2702' 
Local variables:
 self = table: 0x392829d0  {click_offset:table: 0x3941b4d0, children:table: 0x3946d388, ambient_tilt:0.2, T:table: 0x3902b828, offset:table: 0x38deefe0, sell_cost:2, role:table: 0x3901c7d8 (more...)}
 center = nil
 initial = nil
 delay_sprites = nil
 old_center = table: 0x391eb1f8  {alerted:true, _saved_d_u:true, unlocked:true, _u:true, effect:Hand Size Mult, cost:5, order:16, _d:false, blueprint_compat:true, eternal_compat:true (more...)}
(6) Lua global 'create_card_for_shop' at file 'rlSimplify.lua:79' (from mod with id BalatroRL)
Local variables:
 area = table: 0x3901d9c8  {click_offset:table: 0x39b7ac18, static_rotation:false, shuffle_amt:0, parent:table: 0x391c5f20, T:table: 0x39751ee0, offset:table: 0x39142fd0 (more...)}
 forced_tag = nil
 card = table: 0x392829d0  {click_offset:table: 0x3941b4d0, children:table: 0x3946d388, ambient_tilt:0.2, T:table: 0x3902b828, offset:table: 0x38deefe0, sell_cost:2, role:table: 0x3901c7d8 (more...)}
 replace_key = string: "j_smeared_joker"
(7) Lua field 'func' at file 'game.lua:3297'
Local variables:
 nosave_shop = nil
 (for index) = number: 1
 (for limit) = number: 2
 (for step) = number: 1
 i = number: 1
 (*temporary) = Lua function '?' (defined at line 49 of chunk cardarea.lua)
 (*temporary) = table: 0x3901d9c8  {click_offset:table: 0x39b7ac18, static_rotation:false, shuffle_amt:0, parent:table: 0x391c5f20, T:table: 0x39751ee0, offset:table: 0x39142fd0 (more...)}
(8) Lua method 'handle' at file 'engine/event.lua:55'
Local variables:
 self = table: 0x39ce6740  {start_timer:true, timer:TOTAL, blockable:false, trigger:after, func:function: 0x39ce6720, delay:0.2, complete:false, time:476.43000000007, blocking:true (more...)}
 _results = table: 0x393c7e58  {blocking:true, pause_skip:false, time_done:true, completed:false}
(9) Lua method 'update' at file 'engine/event.lua:184'
Local variables:
 self = table: 0x39269818  {queues:table: 0x39269840, queue_last_processed:1234.6166666659, append_count:0, append_queue:base, queue_dt:0.016666666666667, queue_timer:1234.6166666659 (more...)}
 dt = number: 0.0166667
 forced = nil
 (for generator) = C function: next
 (for state) = table: 0x39269840  {unlock:table: 0x39269930, other:table: 0x392699d0, tutorial:table: 0x39269980, base:table: 0x39269958, achievement:table: 0x392699a8}
 (for control) = number: nan
 k = string: "base"
 v = table: 0x39269958  {1:table: 0x39a27180, 2:table: 0x3984bb20, 3:table: 0x39ce6740, 4:table: 0x396357c0, 5:table: 0x39635898, 6:table: 0x39635990, 7:table: 0x390dc058 (more...)}
 blocked = boolean: false
 i = number: 3
 results = table: 0x393c7e58  {blocking:true, pause_skip:false, time_done:true, completed:false}
(10) Lua upvalue 'gameUpdateRef' at file 'game.lua:2632'
Local variables:
 self = table: 0x38ae26b0  {F_GUIDE:false, F_CRASH_REPORTS:false, F_QUIT_BUTTON:true, HUD_tags:table: 0x39701730, F_ENGLISH_ONLY:false, viewed_stake:1, HUD:table: 0x3939e778 (more...)}
 dt = number: 0.0166667
 http_resp = nil
(11) Lua upvalue 'orig_update' at Steamodded file 'src/ui.lua:456' 
Local variables:
 self = table: 0x38ae26b0  {F_GUIDE:false, F_CRASH_REPORTS:false, F_QUIT_BUTTON:true, HUD_tags:table: 0x39701730, F_ENGLISH_ONLY:false, viewed_stake:1, HUD:table: 0x3939e778 (more...)}
 dt = number: 0.0166667
(12) Lua method 'update' at file 'rlSimplify.lua:68' (from mod with id BalatroRL)
Local variables:
 self = table: 0x38ae26b0  {F_GUIDE:false, F_CRASH_REPORTS:false, F_QUIT_BUTTON:true, HUD_tags:table: 0x39701730, F_ENGLISH_ONLY:false, viewed_stake:1, HUD:table: 0x3939e778 (more...)}
 dt = number: 0.0166667
(13) Lua upvalue 'love_update' at file 'main.lua:1024'
Local variables:
 dt = number: 0.0166667
(14) Lua upvalue 'love_update' at file 'src/lua/settings.lua:73' (from mod with id balatrobot)
Local variables:
 _ = number: 0.00391583
(15) Lua field 'update' at file 'balatrobot.lua:76' (from mod with id balatrobot)
Local variables:
 dt = number: 0.00391583
(16) Lua function '?' at file 'main.lua:962' (best guess)
(17) global C function 'xpcall'
(18) LÖVE function at file 'boot.lua:377' (best guess)
Local variables:
 func = Lua function '?' (defined at line 933 of chunk main.lua)
 inerror = boolean: true
 deferErrhand = Lua function '(LÖVE Function)' (defined at line 348 of chunk [love "boot.lua"])
 earlyinit = Lua function '(LÖVE Function)' (defined at line 355 of chunk [love "boot.lua"])
