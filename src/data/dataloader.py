from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def get_batch(
    split: str,
    data_dir: Path,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    path = data_dir / f"{split}.bin"
    data = np.memmap(str(path), dtype=np.uint16, mode="r")

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])

    x, y = x.to(device), y.to(device)
    return x, y
