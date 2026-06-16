import os

from aiwf.runtime.bootstrap_env import apply_runtime_env, parse_perf_argv


def test_parse_perf_argv_defaults_on():
    flags = parse_perf_argv([])
    assert flags.async_offload is True
    assert flags.pinned_memory is True
    assert flags.cuda_malloc is True


def test_parse_perf_argv_disable_flags():
    flags = parse_perf_argv(["--no-async-offload", "--no-pinned-memory", "--no-cuda-malloc"])
    assert flags.async_offload is False
    assert flags.pinned_memory is False
    assert flags.cuda_malloc is False


def test_apply_runtime_env_sets_cuda_malloc(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    applied = apply_runtime_env(cuda_malloc=True)
    assert "cudaMallocAsync" in applied
    assert "backend:cudaMallocAsync" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")