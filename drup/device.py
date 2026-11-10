"""Compute device of the experiment scripts (--device cpu | cuda, default cpu).

Only the model computations move to the device: data loading, splitting, sampling (the CPU generators), model
selection, metrics and the JSON output stay on the CPU. Moving a tensor to the device it already lives on is a no-op,
so the CPU path is unchanged.
"""

import torch


def get_device(name="cpu"):
    """torch.device of a --device value. cuda needs an available CUDA device (checked before any data is read)."""
    if name not in ("cpu", "cuda"):
        raise ValueError(f"unknown device {name!r}")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda: no CUDA device is available in this process")
    return torch.device(name)
