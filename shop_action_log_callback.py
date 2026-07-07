"""
shop_action_log_callback.py — Log shop action frequencies to TensorBoard
========================================================================
Actions are MultiDiscrete [shop_action, strategy]: component 0 is the
shop decision (0=skip, 1-4=buy, 5-9=sell), component 1 is the declared
strategy (0=FLUSH_BUILD, 1=PAIR_BUILD, 2=MULT_BUILD).
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from strategy import NUM_STRATEGIES, STRATEGY_NAMES, Strategy


class ShopActionLogCallback(BaseCallback):
    """Log shop action and strategy frequencies to TensorBoard."""

    def __init__(self, verbose=0, n_actions=10):
        super().__init__(verbose)
        self.action_counts   = [0] * n_actions
        self.strategy_counts = [0] * NUM_STRATEGIES

    def _on_step(self) -> bool:
        actions = self.locals.get("actions", [])
        for a in actions:
            arr = np.asarray(a).ravel()
            shop = int(arr[0])
            if 0 <= shop < len(self.action_counts):
                self.action_counts[shop] += 1
            if len(arr) > 1:
                strat = int(arr[1])
                if 0 <= strat < NUM_STRATEGIES:
                    self.strategy_counts[strat] += 1

        total = sum(self.action_counts)
        if total > 0 and total % 1000 < len(actions):
            self.logger.record("shop/skip_pct", 100 * self.action_counts[0] / total)
            for i in range(1, 5):
                self.logger.record(f"shop/buy_{i}_pct", 100 * self.action_counts[i] / total)
            for i in range(5, len(self.action_counts)):
                self.logger.record(f"shop/sell_{i-4}_pct", 100 * self.action_counts[i] / total)

            strat_total = sum(self.strategy_counts)
            if strat_total > 0:
                for s in Strategy:
                    name = STRATEGY_NAMES[s].lower().replace(" ", "_")
                    self.logger.record(
                        f"strategy/{name}_pct",
                        100 * self.strategy_counts[int(s)] / strat_total,
                    )

        return True
