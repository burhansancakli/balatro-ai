"""Jackdaw CLI — top-level entry point.

Usage::

    jackdaw validate                          # run all scenarios
    jackdaw validate --category jokers        # run only joker scenarios
    jackdaw validate --scenario joker_joker   # run one scenario
    jackdaw validate --host 127.0.0.1 --port 12346

    jackdaw run --mode watch                  # agent plays, you observe
    jackdaw run --mode play                   # you play via browser
    jackdaw run --mode watch --agent greedy --seed SEED42 --deck b_red
    jackdaw run --mode watch --no-web         # terminal only
    jackdaw run --mode watch --no-terminal    # browser only
    jackdaw run --mode watch --host 127.0.0.1 # live mode against real Balatro
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jackdaw",
        description="Headless Balatro simulator for reinforcement learning research",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- validate ------------------------------------------------------------
    p_validate = sub.add_parser(
        "validate",
        help="Validate engine against live Balatro via scenario-based testing",
    )
    p_validate.add_argument(
        "--category",
        choices=("jokers", "tarots", "planets", "spectrals", "boss_blinds", "modifiers", "tags"),
        default=None,
        help="Run only scenarios in this category",
    )
    p_validate.add_argument(
        "--scenario",
        default=None,
        help="Run a single scenario by name",
    )
    p_validate.add_argument(
        "--host",
        default="127.0.0.1",
        help="Balatrobot host (default: 127.0.0.1)",
    )
    p_validate.add_argument(
        "--port",
        type=int,
        default=12346,
        help="Balatrobot port (default: 12346)",
    )
    p_validate.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Seconds between actions for animation watching (default: 0.3)",
    )

    # -- run -----------------------------------------------------------------
    p_run = sub.add_parser(
        "run",
        help="Run the simulator with live visualization (terminal + browser dashboard)",
    )
    p_run.add_argument(
        "--mode",
        choices=("watch", "play"),
        default="watch",
        help="watch = agent plays automatically | play = you control via browser (default: watch)",
    )
    p_run.add_argument(
        "--agent",
        choices=("random", "greedy", "smart", "ppo"),
        default="random",
        help="Agent to use in watch mode (default: random)",
    )
    p_run.add_argument(
        "--model",
        default=None,
        metavar="PATH",
        help="Path to a saved MaskablePPO .zip file (required when --agent ppo)",
    )
    p_run.add_argument(
        "--deck",
        default="b_red",
        metavar="DECK_KEY",
        help="Deck back key, e.g. b_red, b_blue, b_ghost (default: b_red)",
    )
    p_run.add_argument(
        "--stake",
        type=int,
        default=1,
        choices=range(1, 9),
        metavar="1-8",
        help="Stake level 1–8 (default: 1)",
    )
    p_run.add_argument(
        "--seed",
        default=None,
        metavar="SEED",
        help="RNG seed string, e.g. SEED42 (default: random)",
    )
    p_run.add_argument(
        "--speed",
        type=float,
        default=0.8,
        metavar="SECONDS",
        help="Seconds between steps in watch mode (default: 0.8)",
    )
    p_run.add_argument(
        "--no-terminal",
        dest="terminal",
        action="store_false",
        default=True,
        help="Disable Rich terminal output",
    )
    p_run.add_argument(
        "--no-web",
        dest="web",
        action="store_false",
        default=True,
        help="Disable browser dashboard",
    )
    p_run.add_argument(
        "--web-port",
        type=int,
        default=8765,
        metavar="PORT",
        help="Port for the web dashboard (default: 8765)",
    )
    p_run.add_argument(
        "--host",
        default=None,
        metavar="HOST",
        help="Connect to live Balatro via balatrobot (e.g. 127.0.0.1)",
    )
    p_run.add_argument(
        "--port",
        type=int,
        default=12346,
        metavar="PORT",
        help="Balatrobot port (default: 12346, used with --host)",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        from cli.validate import run_validate

        sys.exit(
            run_validate(
                category=args.category,
                scenario=args.scenario,
                host=args.host,
                port=args.port,
                delay=args.delay,
            )
        )

    if args.command == "run":
        _run_command(args)


def _run_command(args: argparse.Namespace) -> None:
    import random as _random
    import time
    import webbrowser

    from visualization.observer import GameObserver

    # Resolve seed
    seed = args.seed or f"SEED{_random.randint(1000, 9999)}"

    # Build adapter for live mode if --host is provided
    adapter = None
    live_mode = args.host is not None
    if live_mode:
        from bridge.backend import LiveBackend
        from emulator.env.game_interface import BridgeAdapter

        backend = LiveBackend(host=args.host, port=args.port)
        adapter = BridgeAdapter(backend)
        adapter.reset(back_key=args.deck, stake=args.stake, seed=seed)
        print(f"  🔴 LIVE MODE — connected to {args.host}:{args.port}")

    observer = GameObserver(
        use_terminal=args.terminal,
        use_web=args.web,
        speed=args.speed,
        port=args.web_port,
    )

    with observer:
        if args.mode == "play":
            if not args.web:
                print("Error: play mode requires the web dashboard (remove --no-web).")
                sys.exit(1)
            url = f"http://127.0.0.1:{args.web_port}/?mode=play"
            print(f"  Seed: {seed}  |  Deck: {args.deck}  |  Stake: {args.stake}")
            print(f"  Opening browser → {url}\n")
            webbrowser.open(url)
            result = observer.simulate_play(
                back_key=args.deck,
                stake=args.stake,
                seed=seed,
                adapter=adapter,
            )
        else:
            # Watch mode
            from emulator.engine.runner import greedy_play_agent, random_agent

            if args.agent == "ppo":
                if not args.model:
                    print("Error: --agent ppo requires --model <path/to/model.zip>")
                    sys.exit(1)
                from cli.ppo_agent import make_ppo_agent
                agent = make_ppo_agent(args.model)
            elif args.agent == "greedy":
                agent = greedy_play_agent
            elif args.agent == "smart":
                from cli.smart_agent import smart_agent
                agent = smart_agent
            else:
                agent = random_agent
            if args.web:
                url = f"http://127.0.0.1:{args.web_port}/?mode=watch"
                print(f"  Opening browser → {url}\n")
                webbrowser.open(url)

            mode_label = "LIVE" if live_mode else "SIM"
            print(f"  Seed: {seed}  |  Deck: {args.deck}  |  Stake: {args.stake}  |  Agent: {args.agent}  |  Mode: {mode_label}\n")
            result = observer.simulate_watched(
                back_key=args.deck,
                stake=args.stake,
                seed=seed,
                agent=agent,
                adapter=adapter,
            )

        # Final summary (inside `with` so the web server is still alive)
        won = result.get("won", False)
        rounds = result.get("round", 0)
        actions = result.get("actions_taken", 0)
        dollars = result.get("dollars", 0)
        status = "WON 🏆" if won else "GAME OVER"
        print(f"\n  {status}  ·  Rounds: {rounds}  ·  Actions: {actions}  ·  Final $: {dollars}\n")

        if args.web:
            print("  Dashboard is still open — press Ctrl+C to close.\n")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n  Closing dashboard.")


if __name__ == "__main__":
    main()
