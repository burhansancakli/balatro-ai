import re
import sys
from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback


class GameStatusCallback(BaseCallback):
    """Display one live status line per seed using tqdm."""

    def __init__(self, verbose=0, total_steps: int = 100_000):
        super().__init__(verbose)
        self.status_by_seed = {}
        self.seed_bars = {}
        self.progress_bar = None
        self.total_steps = total_steps
        self.last_timesteps = 0

    def _format_postfix(self, msg: str) -> str:
        return msg.replace("\n", " ")

    def _on_training_start(self) -> None:
        try:
            seeds = self.training_env.get_attr("seed")
        except Exception:
            seeds = []

        if seeds and sys.stdout.isatty():
            for seed in seeds:
                bar = tqdm(
                    total=1,
                    position=len(self.seed_bars),
                    bar_format="{desc} {postfix}",
                    leave=True,
                    dynamic_ncols=False,
                    ncols=120,
                    disable=False,
                )
                bar.set_description_str(f"[seed {seed}]")
                bar.set_postfix_str(self._format_postfix("waiting..."))
                bar.n = 1
                bar.refresh()
                self.seed_bars[seed] = bar

            self.progress_bar = tqdm(
                total=self.total_steps,
                position=len(self.seed_bars),
                desc="Training",
                leave=True,
                dynamic_ncols=False,
                ncols=120,
                disable=False,
            )
            self.last_timesteps = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        updated = False

        for info in infos:
            jokers = info.get("joker_labels", [])
            joker_str = ", ".join(jokers) if jokers else "none"

            for line in info.get("hand_logs", []):
                match = re.match(r"\[seed ([^\]]+)\] (.*)", line)
                if match:
                    seed = match.group(1)
                    msg = match.group(2)
                    full_msg = f"{msg}  |  jokers=[{joker_str}]"
                    self.status_by_seed[seed] = full_msg
                    updated = True

        if self.progress_bar is not None:
            current = self.num_timesteps
            delta = current - self.last_timesteps
            if delta > 0:
                self.progress_bar.update(delta)
                self.last_timesteps = current

        if updated:
            for seed, msg in self.status_by_seed.items():
                bar = self.seed_bars.get(seed)
                if bar is not None:
                    bar.set_postfix_str(self._format_postfix(msg))
                    bar.refresh()

        return True

    def _on_training_end(self) -> None:
        if self.progress_bar is not None:
            self.progress_bar.close()
        for bar in self.seed_bars.values():
            bar.close()
