import os
import sys

from aiwf.runtime.bootstrap_env import apply_from_argv

apply_from_argv(sys.argv[1:])
os.environ.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from aiwf.app_modern import main

if __name__ == "__main__":
    main()
