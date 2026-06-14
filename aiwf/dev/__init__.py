"""Development-only diagnostics — safe to import; tracing is a no-op when disabled."""

from aiwf.dev.diagnostics import DevDiagnostics, install_dev_diagnostics, trace_model_throughput, trace_safe

__all__ = ["DevDiagnostics", "install_dev_diagnostics", "trace_model_throughput", "trace_safe"]
