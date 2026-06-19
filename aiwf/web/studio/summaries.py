from __future__ import annotations

import re

from aiwf.core.model_profile import detect_model_profile


def result_summary_markdown(job, new_seed, job_status):
    """Format the post-generation status summary shown in Studio."""
    req = job.request
    res = job.result
    if new_seed >= 0:
        head = f"**Done** \u2014 seed **{new_seed}**"
    elif job_status.startswith("**"):
        head = job_status
    else:
        head = f"**Done** \u2014 {job_status}"
    lines = [head]

    prompt = (req.prompt or "").strip().replace("\n", " ")
    if prompt:
        lines.append("_" + (prompt if len(prompt) <= 90 else prompt[:87] + "\u2026") + "_")

    bits = [f"{req.steps} steps", f"CFG {req.cfg_scale:g}", str(req.sampler)]
    sched = getattr(req, "scheduler", "automatic")
    if sched and sched != "automatic":
        bits.append(str(sched))
    bits.append(f"{req.width}\u00d7{req.height}")
    if getattr(req, "batch_size", 1) * getattr(req, "batch_count", 1) > 1:
        bits.append(f"{req.batch_size}\u00d7{req.batch_count} batch")
    lines.append(" \u00b7 ".join(bits))

    elapsed = float(getattr(res, "elapsed_seconds", 0.0) or 0.0)
    if elapsed > 0:
        total_steps = max(1, int(req.steps)) * max(1, len(res.images))
        speed = total_steps / elapsed
        unit = f"{speed:.2f} it/s" if speed >= 1 else f"{1.0 / speed:.2f} s/it"
        lines.append(f"\u23f1 {elapsed:.1f}s \u00b7 {unit}")

    loras = list(dict.fromkeys(re.findall(r"<lora:([^:>]+)", req.prompt or "")))
    if loras:
        lines.append("LoRA: " + ", ".join(loras))

    return "  \n".join(lines)


def model_help_markdown(ckpt_title):
    """One-line guidance for the selected model profile."""
    prof = detect_model_profile(ckpt_title)
    if prof.is_distilled:
        return (
            f"**{prof.title}** \u2014 {prof.help_text}  \n"
            f"Suggested: CFG **{prof.recommended_cfg:g}**, **{prof.recommended_steps}** steps, "
            f"sampler **{prof.recommended_sampler}**, schedule **{prof.recommended_scheduler}**."
        )
    return f"**{prof.title}** \u2014 {prof.help_text}"
