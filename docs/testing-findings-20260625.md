# AIWF Studio — Live Testing Findings (2026-06-25)

Testing performed by driving the actual running app (`webui.bat` → http://localhost:7860) through
the browser UI, with Chrome devtools (console + network) monitored in parallel. App version state
at launch: Python 3.10.9, Torch 2.6.0, Attention=SDP (header badge), Device=RTX 4070 Ti SUPER 16GB,
78 checkpoints loaded, sage attention confirmed active in `aiwf.log` (`DiT attention backend: sage
on Flux2Transformer2DModel`).

## Issue 1 — Image generation hangs indefinitely at "Encoding prompt" (Flux.2 Klein GGUF, CPU offload)

**Severity:** High — job never completes, never errors, never times out.

**Repro:**
1. Launch app fresh via `webui.bat`.
2. Image tab, default loaded checkpoint `fluxtraitFLUX2KleinFLUXZ_klein9bV2Q4KM` (Flux.2 Klein, GGUF
   Q4KM quant), default settings.
3. Prompt: "a red fox sitting in a snowy forest, photorealistic, soft lighting".
4. Click Generate.

**Observed:**
- `aiwf.log` shows the job queue/start sequence normally (`job.queued` → `job.started` → GPU lock
  acquired → `generation.before` → `optimization.profile_resolved` → "Using warm model" →
  "Encoding prompt") at 08:28:32.
- No further log line appears for at least 7 minutes. File size frozen at 854.2KB (verified via
  repeated reads — not a stale-read artifact).
- The Studio UI status panel correctly mirrors this: it stays on "Encoding prompt" the whole time
  (this is NOT a frontend/backend desync — the UI was accurately reporting the stuck state).
- No exception, no OOM message, no `job.failed` — silent stall, not a crash.

**Context that may be relevant:** this same checkpoint OOM-crashed the previous evening
(2026-06-24 21:33) with `CUDAMallocAsyncAllocator` INTERNAL ASSERT FAILED during VAE decode, and
the log shows "Flux.2 Klein on a <10 GB GPU — enabling model CPU offload automatically" — i.e. this
model runs with automatic CPU offload on this 16GB card (odd, since 16GB should not trigger a
<10GB-GPU codepath — worth checking the VRAM-detection threshold logic). Prompt encoding likely
runs the T5/CLIP text encoder on CPU under offload, which could be the root cause of an extreme
slowdown, but 7+ minutes with zero CPU/GPU log feedback and zero progress signal to the user is a
bug regardless of root cause — there is no timeout, watchdog, or user-facing "this may take a while"
messaging.

**Suggested follow-up:** add a stall/timeout watchdog around the text-encode step with a surfaced
warning, and double check why a 16GB card is being routed into the "<10GB GPU" CPU-offload branch
for Flux.2 Klein.

## Issue 2 — "Stop" button does not actually cancel a hung/running generation job

**Severity:** High — combined with Issue 1, there is no way to recover from a stalled job except
restarting the whole app process.

**Repro:** while the job from Issue 1 was stuck at "Encoding prompt" (8+ minutes in), clicked
**Stop** in the Image tab.

**Observed:** `aiwf.log` recorded only a client telemetry line —
`Browser event stop_click: stop_click` — with no corresponding backend action: no `job.cancelled`,
no GPU lock release, no change in job state. The job remained stuck afterward (verified by waiting
20s post-click with no new log activity). The Stop button appears to be purely cosmetic/telemetry
on the frontend and is not wired to an actual cancellation call on the backend job runner.

**Console/network:** no JS errors were thrown when clicking Stop (checked via Chrome devtools
console) — this is a silent no-op, not a crash, which makes it more likely to go unnoticed by a
user who will just assume the app is "still working."

**Suggested follow-up:** wire Stop to a real job-cancellation endpoint that can interrupt a
streaming worker thread/process, and have it actually release the GPU tenant lock so other jobs
(image or video) aren't blocked behind a dead job.

**Refinement:** the frontend is not a pure no-op — the accessibility tree on the Image tab shows
the Stop control transitions to a state literally labeled "— interrupt requested" / "Stopping"
(elements present in the DOM after the click). So the UI *does* track an interrupt request and is
waiting on a backend acknowledgment that never arrives, leaving it stuck permanently in "Stopping"
rather than returning to "Ready" or showing a failure. This is a more precise description than
"purely cosmetic" — the bug is that the interrupt request has no backend handler/ack, not that the
click does nothing client-side.

## Issue 3 — Zombie job from Issue 1 permanently blocks the GPU tenant lock; Video generation cannot start

**Severity:** High — confirms Issues 1 and 2 have a real cross-pipeline consequence: once any job
hangs, the entire app's GPU-bound features (image AND video) are blocked until the process is
restarted.

**Repro:**
1. With the Issue 1 job still hung at "Encoding prompt" (~19 minutes in, never cancelled — see
   Issue 2), switched to the Video tab.
2. Uploaded a source image, selected "Fast: 5B TI2V" route, entered prompt "the camera slowly pans
   across the scene", clicked **Generate Video**.

**Observed:**
- `aiwf.log`: `[GPU] video (job wan_9a816a94) blocked: GPU owned by image (job
  ed3e82e3-...)` — the tenant lock system correctly detects the conflict and refuses to start the
  video job (no crash, no silent overwrite of the image job — that part works as designed).
- The UI does surface an error toast: "GPU busy — GPU is currently owned by image (job ed3e...).
  video blocked. Options: wait for the current job to finish, stop it, or cancel." This is good,
  user-facing messaging — better than Issues 1/2's silence.
- However: there is no actual way to act on either suggested option. "Wait for the current job to
  finish" is not viable because the image job is permanently hung (Issue 1). "Stop it" is not viable
  because Stop does not release the lock (Issue 2). So the toast offers two options, neither of
  which works, with no third option (e.g. "force-release stuck lock") surfaced anywhere in the UI.
- No further log activity for the video job after the blocked message — it does not retry, queue,
  or time out; it simply never starts.
- Browser console showed only pre-existing "Method not implemented" warnings unrelated to this flow
  — no new JS errors thrown by the blocked-job toast itself.

**Suggested follow-up:** add an admin/recovery action (e.g. "Force release GPU lock") reachable from
the blocked-job toast or Settings, for when a tenant lock is held by a job that Stop cannot reach.
This is the practical, user-facing fix until Issue 2's root cause (Stop not wired to real
cancellation) is resolved.

## Verification pass (2026-06-25, post-fix)

After implementing fixes, the app was relaunched fresh (`webui.bat`) and the same repro steps were
re-run live in the browser, with `aiwf.log` monitored in parallel.

**Fixes applied:**
- **Issue 1 (silence):** added a heartbeat log line — `Flux.2 Klein: starting prompt encode (CPU
  offload active — first move of the Qwen3 text encoder to GPU can take a while)` — emitted the
  moment encoding starts, plus a completion line with elapsed seconds. The underlying slowness of
  the Qwen3 text-encoder move under CPU offload is **not** itself fixed (that's expected,
  architecture-driven behavior on a 16GB card) — but it is no longer silent.
- **VRAM-threshold cosmetic bug:** confirmed fixed — log now correctly reads "Flux.2 Klein on a
  **<20 GB** GPU" (was hardcoded to a misleading "<10 GB" string before).
- **Issue 2 (Stop not wired):** `should_cancel` is now threaded through to the three
  architecture-specific encode call sites in `backend.py`, with checks bracketing each encode call.
  This makes Stop effective at the boundaries around encode/load steps. Caveat: if Stop is clicked
  while *inside* the actual blocking encode call (as in this repro, where the call itself is slow),
  it cannot interrupt mid-call — Python has no safe primitive to preempt a thread stuck in a
  synchronous call. The new stall watchdog (below) is the real safety net for that case.
- **Issue 3 (zombie job blocks GPU):** added a stall watchdog that force-fails a job and
  force-releases the GPU tenant lock after 240s of zero progress.

**Re-repro result:** submitted the same prompt ("a red fox sitting in a snowy forest,
photorealistic, soft lighting") against the same checkpoint. The encode stalled again (this is a
real, reproducible slow/stuck path in the Qwen3 CPU-offload encode on this card — not a test
artifact). This time:
1. The heartbeat log appeared immediately (`15:14:51,617`), instead of zero output.
2. At `15:18:52` (241s after the encode started — matching `STALL_TIMEOUT_SECONDS = 240`), the
   watchdog fired: `generation.stall_watchdog_triggered`, force-failed the job
   (`queue.force_failed`), released the GPU tenant lock (`Tenant release: image -> idle`), and the
   UI updated from a permanently-stuck "Encoding prompt" state to a clear error banner: "Generation
   stalled — no progress for 241s ... Forcing this job to fail so the GPU is not blocked
   indefinitely." The Generate button returned to a ready/clickable state — no app restart needed.
3. Confirmed via log that the tenant lock returned to `idle`, meaning a subsequent image or video
   job is free to acquire the GPU — Issue 3's permanent-block scenario no longer requires killing
   the process.

**Known tradeoff (by design, not yet a clean fix):** the original worker thread executing the slow
encode call may still be running in the background even after the watchdog force-fails the job —
Python has no mechanism to forcibly kill a thread blocked in a synchronous call. This is harmless to
the rest of the app: the job queue and GPU tenant lock both guard against a "late return" from an
already-force-failed job overwriting newer state (`_abandoned` set in `queue.py`, plus the tenant
lock's existing owner-id check). The orphaned thread will eventually finish on its own and its
result is discarded. The architecturally clean fix — moving image generation into its own subprocess
with a hard kill, matching the existing Wan/kohya/ed2 pattern — is a larger change than was in scope
here; flagging it as a good candidate for a future pass if these stalls turn out to be frequent.

## Follow-up pass (2026-06-25, later) — log error cleanup + image-mode audit

Triggered by "I still see log errors" + "make every image mode work reliably." Re-read `aiwf.log`
in full and cross-referenced every ERROR/Traceback back to 2026-06-18.

**Fixed this pass:**
- **Orphaned-thread crash after stall force-fail** (`thread.uncaught: Cannot borrow Image
  generation: active owner is none`): the watchdog-abandoned worker thread from the known tradeoff
  above doesn't just discard its result — it finishes the whole generation on real GPU time, then
  crashes calling `borrow_active_tenant()` against a tenant lock already released. Added
  `JobQueue.is_abandoned()` and early-return guards in both the streaming and non-streaming workers
  in `generation.py` so a late-returning forced-failed job discards quietly instead of touching the
  released lock.
- **Normal Stop/watchdog cancellations logged as ERROR tracebacks**: `backend.py`'s generic
  exception handler now has a dedicated `except GenerationCancelledError` clause that logs at INFO
  instead of ERROR with a full traceback.
- **Flux/Flux.2 Klein/Z-Image: "Height/Width must be divisible by 16" ValueError**: these
  architectures patchify in 16x16 blocks; `generate()` was passing `request.width/height` straight
  through with no alignment (SD/SDXL's `align_to_multiple_of_8` helper isn't strict enough and
  wasn't even applied to the base txt2img path). Added `align_to_multiple_of_16` in `mask.py` and
  wired it into the transformer-architecture branch of the per-batch dispatch loop, so e.g. a
  1000x1000 request now silently rounds to 1008x1008 instead of crashing.

**Already fixed by existing code (confirmed via Read, not a live repro — these were historical log
entries from 06-23/06-24 that no longer match current source):**
- `AttributeError: 'ZImagePipeline' object has no attribute 'do_classifier_free_guidance'` —
  current `_encode_z_image_prompts` already uses `getattr(pipe, "do_classifier_free_guidance",
  True)`.
- `AttributeError: 'DiffusersBackend' object has no attribute '_load_dit_transformer_single_file'`
  — method exists today with two call sites; likely a hot-reload-timing artifact at the time.
- `AttributeError: 'Flux2Attention' object has no attribute 'spatial_norm'` — `attention.py` already
  has `ensure_flux2_attention_processors()`, which restores the native Flux2AttnProcessor whenever a
  generic AttnProcessor2_0 swap would otherwise hit this exact attribute. The 06-25 verification
  pass already ran this exact architecture clean through 4 full steps without it recurring.
- `RuntimeError: generator didn't stop` (from `attention_call_context`'s `with` block) — current
  `attention_call_context` is a clean single-yield generator per branch with no loop, which
  structurally can't produce this error; looks like it was already simplified/fixed since 06-24.

**Not yet addressed (flagged, not fixed — lower priority / needs more repro):**
- Recurring frontend JS `TypeError: Cannot read properties of undefined (reading 'flatMap')` (seen
  repeatedly 06-23) — frontend bug, not yet scoped.
- `CUDAMallocAsyncAllocator INTERNAL ASSERT FAILED` (06-24, OOM-adjacent) during VAE decode for
  Flux.2 Klein under CPU offload.
- `KeyError: 'double_blocks.0.img_attn.norm.query_norm.scale'` warmup failure for
  `fluxedUpFluxNSFW_110FP8` — checkpoint-specific key-mapping mismatch.
- `NotImplementedError: Cannot copy out of meta tensor` warmup failure for
  `fluxFusionV24StepsGGUFNF4_V2GGUFQ4KM`.
- "NVIDIA RTX VSR Image Upscale failed" — separate from the core generation pipeline.

**Scope note:** Flux, Flux.2 Klein, and Z-Image are *intentionally* restricted to txt2img only by
design in `generate()` — img2img/inpaint/ControlNet/hires-fix/SDXL-refiner/external-VAE/LoRAs all
raise a clean `ValueError` for these architectures rather than crashing. This is existing,
deliberate scope, not a bug — flagging in case "every image mode" was meant to include extending
these architectures beyond txt2img, which would be a feature addition rather than a fix.

## Note on a suspected Issue (retracted): "Video tab won't switch"

During testing, repeated clicks on the Video nav tab (via element ref, and later via raw screen
coordinates) appeared to leave the panel stuck on "Image." This was investigated as a possible
navigation bug but turned out to be tooling error on the testing side, not an app defect: pixel
coordinates read from a full-desktop screenshot (taken via the computer-use tool) do not share the
same coordinate space as the browser-tab screenshot used to drive clicks (via the Chrome MCP), and
a screen-share banner present only in the desktop capture shifted the apparent layout further. Once
clicks were re-issued using coordinates/refs taken from a Chrome-MCP-native screenshot of the same
tab, the Video tab switched correctly on the first try. No app bug here — noted only so this isn't
mistaken for a finding if retested.

