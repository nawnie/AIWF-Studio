# AIWF Studio Second GUI

Static Windows-friendly preview shell for the screenshot-driven second AIWF Studio interface.

Run from the repository root:

```bat
second_gui.bat
```

Or:

```powershell
.\scripts\run_second_gui.ps1
```

Run against the main AIWF backend:

```powershell
.\scripts\run_second_gui.ps1 -Backend http://127.0.0.1:7860 -Proxy
```

The bridge now wires catalog, progress, status, interrupt, and txt2img proxy behavior when the backend is available. The shell is intentionally safe: missing backend features are marked `WIP` instead of pretending to work.
