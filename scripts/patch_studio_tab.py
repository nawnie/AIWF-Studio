"""Apply modular delegations to studio/tab.py."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "aiwf/web/studio/tab.py"
text = path.read_text(encoding="utf-8")

# Dedent function body (was nested inside register_studio/build)
lines = text.splitlines()
out: list[str] = []
in_build = False
for line in lines:
    if line.startswith("def build_studio_tab"):
        in_build = True
        out.append(line)
        continue
    if in_build and line.startswith("        "):
        out.append(line[4:])
    else:
        out.append(line)
text = "\n".join(out) + "\n"

catalog_block = """    service = ctx.generation
    catalogs = StudioCatalogs.from_context(ctx)
    session = StudioSession()
    runner = GenerationRunner(ctx, service, catalogs, session)
    samplers = service.list_samplers()
    vaes = service.list_vaes()
    vae_choices = [("Automatic", None)] + [(v.title, v.id) for v in vaes]
    default_sampler_label = catalogs.default_sampler_label
    default_schedule_label = catalogs.default_schedule_label
"""
text = re.sub(
    r"    service = ctx\.generation\n.*?default_schedule_label = schedule_id_to_label\.get\(\n.*?\n    \)\n",
    catalog_block,
    text,
    count=1,
    flags=re.DOTALL,
)

text = text.replace(
    'loop_ctrl = {"active": True}\n    sam_state = {"mask": None}\n    inpaint_session = {"original": None, "mask": None}',
    "",
)

# Remove duplicate apply_mode_ui / on_mode_change - use imports
text = re.sub(
    r"\n    def apply_mode_ui\(.*?\n        \)\n",
    "\n",
    text,
    count=1,
    flags=re.DOTALL,
)

text = re.sub(
    r"\n    def on_mode_change\(.*?\n        \)\n",
    "\n",
    text,
    count=1,
    flags=re.DOTALL,
)

replacements = [
    ("on_mode_change(", "on_mode_change(ctx, "),
    ("apply_mode_ui(", "apply_mode_ui(ctx, "),
    ("loop_ctrl[\"active\"] = False", "session.loop_active = False"),
    ("loop_ctrl[\"active\"] = True", "session.loop_active = True"),
    ("while loop_ctrl[\"active\"]:", "while session.loop_active:"),
    ("if not loop_ctrl[\"active\"]:", "if not session.loop_active:"),
    ("sam_state[\"mask\"]", "session.sam_mask"),
    ("sam_state.get(\"mask\")", "session.sam_mask"),
    ('inpaint_session["original"]', "session.inpaint.original"),
    ('inpaint_session["mask"]', "session.inpaint.mask"),
    ("inpaint_session.get(\"original\")", "session.inpaint.original"),
    ("inpaint_session.get(\"mask\")", "session.inpaint.mask"),
    ("inpaint_session,", "session.inpaint_session,"),
    ("_mode_from_label", "mode_from_label"),
]
for old, new in replacements:
    text = text.replace(old, new)

# Wire mode change to imported helper
text = text.replace(
    "mode_toggle.change(\n        on_mode_change,",
    "mode_toggle.change(\n        lambda *a: on_mode_change(ctx, *a),",
)

path.write_text(text, encoding="utf-8")
print("patched tab.py")