import re
import sys
from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback


class GameStatusCallback(BaseCallback):
    """Display one live status line per env instance using tqdm."""

    def __init__(self, verbose=0, total_steps: int = 100_000):
        super().__init__(verbose)
        self.status_by_idx = {}
        self.instance_bars = {}
        self.progress_bar = None
        self.total_steps = total_steps
        self.last_timesteps = 0

    def _format_postfix(self, msg: str) -> str:
        return msg.replace("\n", " ")

    def _on_training_start(self) -> None:
        try:
            n_envs = self.training_env.num_envs
        except Exception:
            n_envs = 0

        if n_envs and sys.stdout.isatty():
            for idx in range(n_envs):
                bar = tqdm(
                    total=1,
                    position=idx,
                    bar_format="{desc} {postfix}",
                    leave=True,
                    dynamic_ncols=False,
                    ncols=120,
                    disable=False,
                )
                bar.set_description_str(f"[env {idx}]")
                bar.set_postfix_str(self._format_postfix("waiting..."))
                bar.n = 1
                bar.refresh()
                self.instance_bars[idx] = bar

            self.progress_bar = tqdm(
                total=self.total_steps,
                position=n_envs,
                desc="Training",
                leave=True,
                dynamic_ncols=False,
                ncols=120,
                disable=False,
            )
            self.last_timesteps = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for idx, info in enumerate(infos):
            jokers = info.get("joker_labels", [])
            joker_str = ", ".join(jokers) if jokers else "none"
            action_label = info.get("action_label", "?")
            ante = info.get("ante", "?")
            round_num = info.get("round", "?")

            # Use last hand log if any, otherwise build a minimal status
            hand_logs = info.get("hand_logs", [])
            if hand_logs:
                msg = hand_logs[-1]
            else:
                msg = f"ante={ante} round={round_num}"

            full_msg = f"{msg}  |  buy={action_label}  |  jokers=[{joker_str}]"
            self.status_by_idx[idx] = full_msg

        if self.progress_bar is not None:
            current = self.num_timesteps
            delta = current - self.last_timesteps
            if delta > 0:
                self.progress_bar.update(delta)
                self.last_timesteps = current

        for idx, msg in self.status_by_idx.items():
            bar = self.instance_bars.get(idx)
            if bar is not None:
                bar.set_postfix_str(self._format_postfix(msg))
                bar.refresh()

        return True

    def _on_training_end(self) -> None:
        if self.progress_bar is not None:
            self.progress_bar.close()
        for bar in self.instance_bars.values():
            bar.close()