"""Write one test scalar so TensorBoard setup can be verified quickly."""

import os
import time

from torch.utils.tensorboard import SummaryWriter


LOG_DIR = "./logs/tensorboard_probe"


def main() -> int:
    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(LOG_DIR)
    writer.add_scalar("debug/probe_value", 1.0, 0)
    writer.add_scalar("debug/probe_value", 2.0, 1)
    writer.flush()
    writer.close()
    print(f"Wrote TensorBoard probe scalars to {LOG_DIR}")
    print("Open TensorBoard at http://localhost:6006 and look for debug/probe_value.")
    time.sleep(0.1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
