# r/Python Draft

## Title

Managing a multi-venv local AI app from one Python controller without making optional engines boot dependencies

## Draft

I am building AIWF Studio, a local-first AI application where the Python architecture problem has become more interesting than the UI itself.

The project needs to support a stable Gradio/FastAPI core plus optional heavy runtimes:

- Diffusers image generation
- Wan video experiments
- Kohya LoRA training
- EveryDream2 full fine-tuning
- GGUF/Ollama chat integration

The rule I settled on: optional engines must never become mandatory boot dependencies.

That led to a controller pattern:

- `launch.py` owns the main app venv bootstrap.
- `engines.json` declares optional engine venvs.
- `scripts/bootstrap_engine.ps1 <name>` creates isolated engine environments.
- `ProcessSupervisor` launches named subprocess workers with `shell=False`.
- Workers communicate by JSONL events on stdout.
- The UI imports service boundaries, not training libraries.

The repo currently has 715+ collected tests, with 722 collected in my workspace through the project venv:

```powershell
.\venv\Scripts\python.exe -m pytest --collect-only -q tests
```

Contributor setup:

```powershell
git clone https://github.com/nawnie/AIWF-Studio.git
cd AIWF-Studio
python launch.py --skip-sageattention --skip-wan --skip-kohya --skip-ed2
.\venv\Scripts\python.exe -m pytest tests -q
```

Repo:

https://github.com/nawnie/AIWF-Studio

I would be interested in feedback from Python infrastructure developers on the subprocess protocol, setup ergonomics, and test strategy for optional dependency stacks.

## Posting notes

- Re-check r/Python rules before posting.
- Keep this framed around Python architecture, not AI art.
- If self-promotion is restricted, use an allowed showcase or self-promotion thread instead.
