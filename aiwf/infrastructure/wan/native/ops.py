"""Operation factories for AIWF-native Wan modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class WanOps:
    """Default operation factory matching ordinary PyTorch modules."""

    name = "bf16"

    def Linear(self, in_features: int, out_features: int, bias: bool = True, **kwargs: Any):
        import torch

        return torch.nn.Linear(in_features, out_features, bias=bias, **kwargs)

    def Conv2d(self, *args: Any, **kwargs: Any):
        import torch

        return torch.nn.Conv2d(*args, **kwargs)

    def Conv3d(self, *args: Any, **kwargs: Any):
        import torch

        return torch.nn.Conv3d(*args, **kwargs)

    def LayerNorm(self, *args: Any, **kwargs: Any):
        import torch

        return torch.nn.LayerNorm(*args, **kwargs)

    def RMSNorm(self, *args: Any, **kwargs: Any):
        import torch

        if hasattr(torch.nn, "RMSNorm"):
            return torch.nn.RMSNorm(*args, **kwargs)
        return torch.nn.LayerNorm(*args, **kwargs)


class WanBF16Ops(WanOps):
    """Explicit full/bf16 operation factory."""

    name = "bf16"


class WanFP8Ops(WanOps):
    """Quantized Wan operation factory using AIWF FP8 Linear."""

    name = "fp8"

    def Linear(self, in_features: int, out_features: int, bias: bool = True, **kwargs: Any):
        from aiwf.infrastructure.quant.fp8_linear import AIWFFP8Linear

        return AIWFFP8Linear(in_features, out_features, bias=bias, **kwargs)


@dataclass
class WanDiagnosticOps(WanOps):
    """Operation factory that records module construction for smoke tests and diagnostics."""

    base: WanOps = field(default_factory=WanBF16Ops)
    created: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)
    name: str = "diagnostic"

    def Linear(self, *args: Any, **kwargs: Any):
        self.created.append(("Linear", args, dict(kwargs)))
        return self.base.Linear(*args, **kwargs)

    def Conv2d(self, *args: Any, **kwargs: Any):
        self.created.append(("Conv2d", args, dict(kwargs)))
        return self.base.Conv2d(*args, **kwargs)

    def Conv3d(self, *args: Any, **kwargs: Any):
        self.created.append(("Conv3d", args, dict(kwargs)))
        return self.base.Conv3d(*args, **kwargs)

    def LayerNorm(self, *args: Any, **kwargs: Any):
        self.created.append(("LayerNorm", args, dict(kwargs)))
        return self.base.LayerNorm(*args, **kwargs)

    def RMSNorm(self, *args: Any, **kwargs: Any):
        self.created.append(("RMSNorm", args, dict(kwargs)))
        return self.base.RMSNorm(*args, **kwargs)
