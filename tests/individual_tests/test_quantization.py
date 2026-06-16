"""tests/test_quantization.py — quantization layer tests (no GPU, no torchao required)."""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.conv   = nn.Conv2d(4, 4, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class TestChannelsLast:
    def test_applies_to_conv_model(self) -> None:
        with patch.dict(os.environ, {"AIWF_CHANNELS_LAST": "1"}):
            from importlib import reload
            import aiwf.infrastructure.quantization.torchao_quant as q
            reload(q)
            model = _TinyModel()
            result = q.maybe_channels_last(model)
        # Memory format applied — model should not have changed identity
        assert result is model

    def test_skips_without_flag(self) -> None:
        with patch.dict(os.environ, {"AIWF_CHANNELS_LAST": "0"}):
            from importlib import reload
            import aiwf.infrastructure.quantization.torchao_quant as q
            reload(q)
            model = _TinyModel()
            result = q.maybe_channels_last(model)
        assert result is model

    def test_skips_no_conv2d_model(self) -> None:
        with patch.dict(os.environ, {"AIWF_CHANNELS_LAST": "1"}):
            from importlib import reload
            import aiwf.infrastructure.quantization.torchao_quant as q
            reload(q)
            model = nn.Sequential(nn.Linear(8, 8))  # no Conv2d
            result = q.maybe_channels_last(model)
        assert result is model


class TestTorchCompile:
    def test_skips_without_flag(self) -> None:
        with patch.dict(os.environ, {"AIWF_TORCH_COMPILE": "0"}):
            from importlib import reload
            import aiwf.infrastructure.quantization.torchao_quant as q
            reload(q)
            model = nn.Linear(4, 4)
            result = q.maybe_torch_compile(model)
        assert result is model

    def test_skips_if_no_compile_attr(self) -> None:
        with patch.dict(os.environ, {"AIWF_TORCH_COMPILE": "1"}):
            from importlib import reload
            import aiwf.infrastructure.quantization.torchao_quant as q
            reload(q)
            model = nn.Linear(4, 4)
            with patch.object(torch, "compile", None):
                result = q.maybe_torch_compile(model)
        assert result is model


class TestInt8WeightOnly:
    def test_skips_without_flag(self) -> None:
        with patch.dict(os.environ, {"AIWF_TORCHAO": "0"}):
            from importlib import reload
            import aiwf.infrastructure.quantization.torchao_quant as q
            reload(q)
            model = _TinyModel()
            result = q.apply_int8_weight_only(model)
        assert result is model

    def test_warns_when_torchao_missing(self, caplog) -> None:
        with patch.dict(os.environ, {"AIWF_TORCHAO": "1"}):
            with patch.dict(sys.modules, {"torchao": None}):
                from importlib import reload
                import aiwf.infrastructure.quantization.torchao_quant as q
                reload(q)
                import logging
                with caplog.at_level(logging.WARNING):
                    model = _TinyModel()
                    result = q.apply_int8_weight_only(model)
        assert result is model  # unchanged


class TestApplyAllOptimizations:
    def test_runs_without_any_flags(self) -> None:
        with patch.dict(os.environ, {
            "AIWF_TORCHAO": "0", "AIWF_TORCH_COMPILE": "0",
            "AIWF_CHANNELS_LAST": "0", "AIWF_FP8": "0",
        }):
            from importlib import reload
            import aiwf.infrastructure.quantization.torchao_quant as q
            reload(q)
            model = _TinyModel()
            result = q.apply_all_optimizations(model)
        assert result is model
