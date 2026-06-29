"""
research_callback.py — Log Balatro game metrics to TensorBoard
==============================================================
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ResearchCallback(BaseCallback):
    """Log Balatro game metrics to TensorBoard, aggregated per episode."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._ep_antes:  list[float] = []
        self._ep_wins:   list[float] = []
        self._ep_rounds: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            if done:
                ante          = int(info.get("ante_reached", info.get("ante", 1)))
                won           = float(info.get("won", False))
                rounds_beaten = max(0, int(info.get("round", 1)) - 1)

                self._ep_antes.append(ante)
                self._ep_wins.append(won)
                self._ep_rounds.append(rounds_beaten)

        return True

    def _on_rollout_end(self) -> None:
        if not self._ep_antes:
            return

        self.logger.record("balatro/mean_ante_reached",  np.mean(self._ep_antes))
        self.logger.record("balatro/max_ante_reached",   float(np.max(self._ep_antes)))
        self.logger.record("balatro/win_rate",           np.mean(self._ep_wins))
        self.logger.record("balatro/mean_rounds_beaten", np.mean(self._ep_rounds))

        self._ep_antes  = []
        self._ep_wins   = []
        self._ep_rounds = []
