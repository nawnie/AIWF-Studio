# Gradio to Pro React Workflow

This note is for agents adding UI-facing features to AIWF Studio.

## Role split

- Python owns behavior. Put new generation, processing, settings, catalog, or job logic in a plain Python module or service first.
- Gradio Studio is the development blueprint. It gets the first UI wiring because it is the broadest working surface and exposes callback bugs quickly.
- Pro is the production-oriented React/FastAPI version. Do not use it for first-pass experiments unless Shawn asks for a direct Pro change.

This matches the existing project split: README documents Pro as `AIWF Studio Pro.bat`, `launch_pro.py`, `aiwf/app_pro.py`, `aiwf/web/pro_api.py`, and `frontend/`. Shared backend and job contracts should flow outward only after Studio QA.

## Standard path

1. Add or change a plain Python file.
   - Keep business logic out of UI files.
   - Define request and result shapes before touching UI controls.
   - Add focused unit coverage around validation, fallback behavior, and missing local assets.

2. Validate the Python path.
   - Use focused `py_compile`, pytest, list-mode, preflight, or no-GUI smoke commands.
   - Do not run large downloads, training, or VRAM-heavy generation unless Shawn asked for it.

3. Add the Gradio element.
   - Put controls near the relevant action, especially persistent Generate-time options.
   - Wire the control to the Python module with clear input ordering.
   - Keep labels short and specific.

4. Validate Gradio and run a small bug pass.
   - Check callback argument order and return shape.
   - Check defaults, disabled states, missing model messages, settings persistence, progress, cancellation, output paths, and error states.
   - Add or update a focused test under `tests/individual_tests/` when the behavior is not already covered.

5. Promote to Pro.
   - Add or align the API contract in `aiwf/web/pro_api.py`.
   - Mirror the proven Gradio controls and states in `frontend/`.
   - Do not copy Gradio UI source into React. Port the behavior and API contract.

6. Validate Pro.
   - Run the focused Pro API tests.
   - Build the frontend.
   - Check for TypeScript type drift between `frontend/src/types.ts`, `frontend/src/api.ts`, and the API response.

## React porting notes

Use the `build-web-apps:react-best-practices` skill for React work. For this repo, the highest-value rules are:

- Start independent API requests together and await them late when a Pro screen needs multiple `/api/pro/*` payloads.
- Keep heavy optional code lazy. Prompt analysis and other browser-side helpers should load only when the user activates them.
- Avoid duplicated business logic in React. React should format, validate user inputs lightly, and call the API contract proven by Python and Gradio.
- Keep TypeScript payload types close to the API shape. Prefer explicit normalizers in `frontend/src/api.ts` over ad hoc parsing in components.
- Use primitive effect dependencies, functional state updates, and stable callbacks for interactive controls.
- Do not define React components inside other components.
- Use direct imports where practical, and avoid broad barrel imports that pull unused code into the Pro bundle.

## Useful validation commands

Choose the focused subset that matches the change:

```powershell
venv\Scripts\python.exe -m py_compile path\to\changed_module.py
venv\Scripts\python.exe -m pytest tests\individual_tests\test_web_registry.py -q
venv\Scripts\python.exe -m pytest tests\individual_tests\test_pro_api.py -q
venv\Scripts\python.exe -c "from aiwf.bootstrap import build_context; from aiwf.web.app import create_web_ui; ctx = build_context(); create_web_ui(ctx); print('ui smoke ok')"
Push-Location frontend; npm run build; Pop-Location
```

For pipeline work, also run the relevant preflight or registry test. For pure docs or agent-guidance edits, a readback plus whitespace check is enough.
