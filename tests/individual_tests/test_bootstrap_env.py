import os

from aiwf.runtime.bootstrap_env import apply_runtime_env, parse_perf_argv


def test_parse_perf_argv_defaults_to_safe_allocator(tmp_path):
    flags = parse_perf_argv(["--data-dir", str(tmp_path)])
    assert flags.async_offload is True
    assert flags.pinned_memory is True
    assert flags.cuda_malloc is False


def test_parse_perf_argv_can_enable_cuda_malloc():
    flags = parse_perf_argv(["--cuda-malloc"])
    assert flags.cuda_malloc is True


def test_parse_perf_argv_reads_saved_cuda_malloc(tmp_path):
    (tmp_path / "launch.json").write_text('{"cuda_malloc": true}', encoding="utf-8")
    flags = parse_perf_argv(["--data-dir", str(tmp_path)])
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


def test_apply_runtime_env_leaves_allocator_unset_when_safe_default(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    applied = apply_runtime_env(cuda_malloc=False)
    assert "cudaMallocAsync" not in applied
    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
