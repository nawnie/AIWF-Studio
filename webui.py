import os

os.environ.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from aiwf.app import main

if __name__ == "__main__":
    main()
