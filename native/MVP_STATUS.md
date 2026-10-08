# AIWF Studio MVP status — 2026-10-08

The native Windows MVP has a fresh Release build and passing headless checks. Acceptance is pending native runtime, visual/accessibility and clean-machine checks. This report covers the current candidate in `F:\AIWF_Studio`; the dated demonstration claims in DESIGN.md are historical evidence.

Scope: Home, Chat, Create, Datasets, training preparation, Projects, Settings and Setup. Studio-wide Audio and Pro workspaces remain part of AIWF Studio. Native training execution/monitoring, MSIX, auto-update and support for every model family are outside this native MVP scope.

Evidence workspace: `C:\Users\Shawn\Documents\Codex\2026-10-08\finish-aiwf-to-mvp-status`. Paths below are relative to that workspace unless marked otherwise.

| Gate | Result | Evidence and limits |
|---|---|---|
| Workflow/API contracts | Pass: 88 tests | `mvp-contract-tests.log`: unified bridge, image jobs, agent access, Audio project and API contracts. Fixture tests; no GPU work. |
| Startup/installer/model readiness | Pass: 92 tests | `mvp-readiness-tests.log`: installer safety, model startup, route lifecycle and model/engine setup contracts. Existing warnings remain. |
| Pro frontend build | Pass | `frontend-build.log`: production build, existing large-chunk warning. Pro source was unchanged in this task. |
| Native Release build | Pass | `native-final-build-v3/build.log`, exit code 0; 108 pinned files in `source-hashes.json`; canonical files match. |
| Native logic regressions | Pass | `native-logic-receipt.json`: eight selected actual method bodies plus the StartAsync registration prefix compiled with doubles and a deterministic scheduler; old Chat, StopAll and pre-fix start-registration negative controls fail as intended. Full async flows, process launching and rendering were not exercised. |
| Native installer | Pass in isolated fixture | `native-install-receipt.json`: outside-root/unmarked-folder guards, exact executable copy and repeat marked install. No app or shortcut was launched. |
| VC runtime prerequisite | Pass in fixtures | `runtime-prerequisite-receipt.json`: missing, partial and present DLL cases. Does not prove clean-machine DLL loading. |
| Independent source review | Accepted within reviewed scope | `repair-review-v3-report.md`; reviewer found no remaining concrete source blocker. Reviewer session `01a11bc3-0bf7-7f32-af3e-50b738b2ea09`, configured `gpt-6.1-sol`, high reasoning, default tier. |
| Current service probes | Partial readiness | `runtime-probes.json`, checked 2026-10-08 13:30 UTC: Qwen Chat/ComfyUI answer; Dataset Studio lists one published package; ReTrain is not running. The temporary host's output root was outside the Dataset catalog allowlist, so that result does not assess the canonical output root. |
| Native end-to-end replay | Pending | No supported native app capture/control interface was available in this task. No native launch or integrated UI replay was performed. |
| Keyboard, Narrator and scaling | Pending | No fresh native rendered/accessibility evidence. |
| Clean-machine runtime | Pending | Requires a prepared Windows machine with the documented checkout, Python environment and x64 VC runtime. The isolated installer fixture is not a clean-machine run. |

Repairs in this task:

- Chat guards New chat during an outstanding request and creates cancellation before project-context loading.
- Train clears imported-package state when the project changes, rejects stale responses and keeps busy controls disabled across status refreshes.
- Engine starts use stop generations, per-start tokens and ownership checks. Stop revokes pending work atomically; stale health/status results cannot replace newer state. StopAll remains available during startup. Existing external services retain external ownership.
- Create refreshes its output root, retains job tracking through connection failures, reports cancellation errors and exposes an explicit dismissal when tracking cannot recover. Dismissal does not claim the remote job was cancelled.
- Home and Projects reject late activity responses for a different project; Projects follows title-bar selection.
- Readiness requires a successful authenticated health response. Dataset token resolution includes saved Setup choices and passes the same token file to native probes and the engine API. Setup reports the required restart for externally owned services.
- The installer checks for the required x64 VC runtime DLLs before copying its payload.

Exact changes and rollback copies: `native-change-receipt.json`, `native-mvp-repairs.patch`, `source-backups`. The native tree already contained extensive unrelated work; this task preserved it. The incremental patch passes a reverse-apply check.

Candidate executable: `F:\AIWF_Studio\native\mvp-candidate-20261008\AIWFStudio.exe`. The installer copied the verified build into a new folder; `candidate-install-receipt.json` checks the payload against the build. The existing install and Desktop shortcut were left in place. The candidate has not been launched.

SHA-256: `7ECB0EA596A56C8CD06D5C934155A7543A2AD753976459EA14DDD437D51BD6BC`.

Acceptance replay still required:

1. Launch the candidate and check all pages, theme, keyboard navigation, Narrator labels and 100/150/200% scaling.
2. Check New chat/Stop during context loading and streaming; change projects during imports and plans; stop individual/batched engines during startup and verify only app-owned processes stop.
3. Check image completion, cancellation, connection loss/API restart and Dismiss tracking; confirm results open from the current output folder.
4. On isolated Dataset/ReTrain fixture data, change the Dataset state folder, restart the appropriate services, import a published revision and obtain a dry-run plan. ReTrain must be available for this gate.
5. Install and launch on a prepared second Windows machine; verify missing prerequisites produce actionable errors and existing data is preserved.

No models were downloaded, no inference or training ran, no real Dataset/ReTrain records were changed, and no online publication occurred. Local AES wiki updates record the evidence and remaining gates.

Shawn Core routed this scope and produced the initial isolated native build. Its bounded execution deadline expired during later build dispatch. The final builds used the same authorized local MSBuild procedure with fresh pinned snapshots; the Core scope was not reset or represented as a passing final build.

Final Core validation returned `accepted: false`, `outcome: stopped`, with the remaining native/runtime gates recorded in `core-final-validation.json`. The independent review accepted source repairs; it did not certify the complete MVP acceptance matrix.


