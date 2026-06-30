from __future__ import annotations

import os
import sys

import launch
from aiwf.runtime.bootstrap_env import apply_from_argv


def main() -> None:
    """Prepare the shared AIWF environment, then launch the Pro React app."""
    argv = sys.argv[1:]
    sys.path.insert(0, str(launch.ROOT))

    apply_from_argv(argv)
    os.environ.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    skip_prepare = "--skip-prepare-environment" in argv
    skip_install = "--skip-install" in argv
    launch.prepare(skip_prepare, skip_install, argv)
    pro_argv = launch.strip_launch_only_args(argv)

    env = os.environ.copy()
    env.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env["PYTHONPATH"] = (
        str(launch.ROOT)
        if "PYTHONPATH" not in env
        else str(launch.ROOT) + os.pathsep + env["PYTHONPATH"]
    )

    os.execvpe(
        launch.python(),
        [launch.python(), str(launch.ROOT / "webui_pro.py"), *pro_argv],
        env,
    )


if __name__ == "__main__":
    main()
