import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE.*")
warnings.filterwarnings("ignore", message=".*cudaMallocAsync ignores max_split_size_mb.*")

from aiwf.runtime.bootstrap_env import apply_from_argv

apply_from_argv(sys.argv[1:])
os.environ.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from aiwf.app import main

if __name__ == "__main__":
    main()
