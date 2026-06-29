"""
shop_action_log_callback.py — Log shop action frequencies to TensorBoard
========================================================================
"""

from stable_baselines3.common.callbacks import BaseCallback


class ShopActionLogCallback(BaseCallback):
    """Log shop action frequencies to TensorBoard."""

    def __init__(self, verbose=0, n_actions=10):
        super().__init__(verbose)
        self.action_counts = [0] * n_actions

    def _on_step(self) -> bool:
        actions = self.locals.get("actions", [])
        for a in actions:
            idx = int(a)
            if 0 <= idx < len(self.action_counts):
                self.action_counts[idx] += 1

        total = sum(self.action_counts)
        if total > 0 and total % 1000 < len(actions):
            self.logger.record("shop/skip_pct", 100 * self.action_counts[0] / total)
            for i in range(1, 5):
                self.logger.record(f"shop/buy_{i}_pct", 100 * self.action_counts[i] / total)
            for i in range(5, len(self.action_counts)):
                self.logger.record(f"shop/sell_{i-4}_pct", 100 * self.action_counts[i] / total)

        return True
