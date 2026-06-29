"""
run_checkpoint_callback.py — CheckpointCallback that auto-names after TensorBoard run
======================================================================================
"""

from pathlib import Path
from stable_baselines3.common.callbacks import CheckpointCallback


class RunCheckpointCallback(CheckpointCallback):
    """CheckpointCallback that auto-names files after the TensorBoard run."""

    def __init__(self, save_freq: int, save_path: str):
        super().__init__(save_freq=save_freq, save_path=save_path, name_prefix="")

    def _on_training_start(self) -> None:
        run_name = Path(self.model.logger.dir).name
        self.name_prefix = run_name
