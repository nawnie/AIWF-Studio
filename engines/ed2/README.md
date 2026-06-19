# ED2 Engine

This folder owns the optional EveryDream2 full fine-tuning engine.

Expected layout:

```text
engines/ed2/
├── worker.py
├── requirements.txt
├── .venv/                    # optional isolated runtime, ignored by git
└── EveryDream2trainer/       # Shawn's ED2 fork or upstream ED2, ignored by git
```

To install Shawn's fork:

```powershell
$env:AIWF_ED2_REPO_URL = "https://github.com/<account>/EveryDream2trainer.git"
python -c "from aiwf.services.training.ed2_installer import install_ed2_addon; print('\n'.join(install_ed2_addon()))"
```

The Training tab installer calls the same code. By default the installer uses
Studio runtime mode and installs only AIWF's ED2 overlay requirements. It does
not install the upstream ED2 requirements into the main Studio venv.
