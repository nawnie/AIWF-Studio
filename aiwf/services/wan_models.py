from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WanModelPairCheck:
    ok: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    high_family: str = ""
    low_family: str = ""
    high_storage: str = ""
    low_storage: str = ""


def wan_model_stage_role(model_id: str | None) -> str:
    """Classify a Wan transformer filename as high/low/unknown."""
    name = Path(str(model_id or "")).name.lower()
    stem = Path(name).stem
    has_high = "high" in stem
    has_low = "low" in stem
    if has_high and not has_low:
        return "high"
    if has_low and not has_high:
        return "low"
    return "unknown"


def wan_model_storage_family(model_id: str | None) -> str:
    suffix = Path(str(model_id or "")).suffix.lower()
    if suffix == ".gguf":
        return "gguf"
    if suffix == ".safetensors":
        return "safetensors"
    if not suffix and model_id:
        return "diffusers_or_folder"
    return "unknown"


def wan_model_quant_family(model_id: str | None) -> str:
    name = Path(str(model_id or "")).name.lower()
    if "fp8" in name or "f8" in name:
        return "fp8"
    for pattern in (
        r"q(\d+)(?=high|low)",
        r"(?:^|[_\-.])q(\d+)(?:[_\-.]|$)",
        r"(?:^|[_\-.])q(\d+)_k(?:[_\-.]|$)",
        r"(?:^|[_\-.])q(\d+)_\d(?:[_\-.]|$)",
    ):
        match = re.search(pattern, name)
        if match:
            return f"q{match.group(1)}"
    if "bf16" in name:
        return "bf16"
    if "fp16" in name:
        return "fp16"
    return "unknown"


def wan_model_pair_family_key(model_id: str | None) -> str:
    name = Path(str(model_id or "")).name.lower()
    stem = Path(name).stem
    stem = re.sub(r"(high|low)[_\-. ]*noise", "", stem)
    stem = stem.replace("high", "").replace("low", "")
    stem = re.sub(r"[_\-. ]+", "_", stem)
    return stem.strip("_")


def wan_model_pair_compatibility(high_model_id: str | None, low_model_id: str | None) -> WanModelPairCheck:
    high_id = str(high_model_id or "").strip()
    low_id = str(low_model_id or "").strip()
    if not high_id or not low_id:
        return WanModelPairCheck(ok=True)

    high_role = wan_model_stage_role(high_id)
    low_role = wan_model_stage_role(low_id)
    high_storage = wan_model_storage_family(high_id)
    low_storage = wan_model_storage_family(low_id)
    high_quant = wan_model_quant_family(high_id)
    low_quant = wan_model_quant_family(low_id)
    high_family = wan_model_pair_family_key(high_id)
    low_family = wan_model_pair_family_key(low_id)

    errors: list[str] = []
    warnings: list[str] = []
    if high_role == "low":
        errors.append(f"High noise selection looks like a low-noise model: {Path(high_id).name}")
    elif high_role == "unknown":
        warnings.append(f"High noise filename does not clearly say high: {Path(high_id).name}")
    if low_role == "high":
        errors.append(f"Low noise selection looks like a high-noise model: {Path(low_id).name}")
    elif low_role == "unknown":
        warnings.append(f"Low noise filename does not clearly say low: {Path(low_id).name}")

    if high_storage != "unknown" and low_storage != "unknown" and high_storage != low_storage:
        errors.append(
            "High and Low noise transformers use different storage formats "
            f"({high_storage} vs {low_storage}). Select a matched pair."
        )

    if high_quant != "unknown" and low_quant != "unknown" and high_quant != low_quant:
        errors.append(
            "High and Low noise transformers use different quantization tiers "
            f"({high_quant} vs {low_quant}). Select matching Q/FP precision files."
        )

    high_size = low_size = 0
    try:
        high_size = Path(high_id).stat().st_size
        low_size = Path(low_id).stat().st_size
    except OSError:
        pass
    if high_size > 0 and low_size > 0:
        smaller = min(high_size, low_size)
        larger = max(high_size, low_size)
        ratio = larger / max(1, smaller)
        if ratio >= 1.35:
            errors.append(
                "High and Low noise transformer file sizes differ too much for a normal matched pair "
                f"({high_size / 1024**3:.2f} GiB vs {low_size / 1024**3:.2f} GiB)."
            )
        elif ratio >= 1.15:
            warnings.append(
                "High and Low noise transformer file sizes differ noticeably "
                f"({high_size / 1024**3:.2f} GiB vs {low_size / 1024**3:.2f} GiB)."
            )

    if errors and os.environ.get("AIWF_WAN_ALLOW_MISMATCHED_PAIR", "").strip().lower() in {"1", "true", "yes", "on"}:
        warnings.extend(f"Ignored by AIWF_WAN_ALLOW_MISMATCHED_PAIR=1: {error}" for error in errors)
        errors = []

    return WanModelPairCheck(
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        high_family=high_family,
        low_family=low_family,
        high_storage=high_storage,
        low_storage=low_storage,
    )


def wan_autopair_counterpart(selected_id, candidates, *, want_role: str):
    """Find the best opposite-stage transformer to pair with ``selected_id``.

    Given a chosen high- (or low-) noise transformer and a list of available
    transformer ids/paths, return the best counterpart whose stage role is
    ``want_role`` ("low" or "high"). Never crosses storage formats (a GGUF high
    pairs only with a GGUF low). Prefers the same pair-family key, then the same
    quantization tier. Returns None if nothing suitable is found.
    """
    sel = str(selected_id or "").strip()
    if not sel or want_role not in ("high", "low"):
        return None
    sel_family = wan_model_pair_family_key(sel)
    sel_storage = wan_model_storage_family(sel)
    sel_quant = wan_model_quant_family(sel)
    sel_name = Path(sel).name
    best = None
    best_score = 0
    for cand in candidates or ():
        cid = str(cand or "").strip()
        if not cid or Path(cid).name == sel_name:
            continue
        if wan_model_stage_role(cid) != want_role:
            continue
        # Never auto-pair across storage formats — a mismatched pair fails to load.
        if sel_storage != "unknown" and wan_model_storage_family(cid) != sel_storage:
            continue
        score = 1  # role (+ storage) already matched
        if sel_family and wan_model_pair_family_key(cid) == sel_family:
            score += 4
        if sel_quant != "unknown" and wan_model_quant_family(cid) == sel_quant:
            score += 2
        if score > best_score:
            best, best_score = cid, score
    return best


def wan_autopair_low_for_high(high_model_id, candidates):
    """Convenience: best low-noise counterpart for a chosen high-noise model."""
    return wan_autopair_counterpart(high_model_id, candidates, want_role="low")


def wan_autopair_high_for_low(low_model_id, candidates):
    """Convenience: best high-noise counterpart for a chosen low-noise model."""
    return wan_autopair_counterpart(low_model_id, candidates, want_role="high")


def wan_vae_generation(vae_id):
    """Infer Wan VAE generation from filename: '2.1' (16-ch) vs '2.2' (48-ch)."""
    n = Path(str(vae_id or "")).name.lower()
    if "2.2" in n or "wan22" in n or "_22" in n:
        return "2.2"
    if "2.1" in n or "wan21" in n or "_21" in n:
        return "2.1"
    return "unknown"


def wan_text_encoder_kind(te_id):
    """Classify a Wan text-encoder file: gguf / fp8 / full / none."""
    n = Path(str(te_id or "")).name.lower()
    if not n:
        return "default"
    if n.endswith(".gguf"):
        return "gguf"
    if "fp8" in n or "f8" in n:
        return "fp8"
    if n.endswith(".safetensors"):
        return "safetensors"
    return "unknown"


def wan_setup_summary(*, runtime_mode="", high_id=None, low_id=None, vae_id=None,
                      text_encoder_id=None, dual=False):
    """Structured pre-launch diagnostics for the Wan UI panel.

    Pure/stdlib — safe to call from the UI or preflight. Returns a dict of detected
    facts plus human-readable ``lines`` and ``warnings`` for display. No file loading
    beyond optional stat() inside the pair check.
    """
    high_id = str(high_id or "").strip()
    low_id = str(low_id or "").strip()
    summary = {
        "runtime_mode": runtime_mode,
        "dual": bool(dual),
        "high": {
            "name": Path(high_id).name if high_id else "",
            "role": wan_model_stage_role(high_id) if high_id else "",
            "storage": wan_model_storage_family(high_id) if high_id else "",
            "quant": wan_model_quant_family(high_id) if high_id else "",
        },
        "low": {
            "name": Path(low_id).name if low_id else "",
            "role": wan_model_stage_role(low_id) if low_id else "",
            "storage": wan_model_storage_family(low_id) if low_id else "",
            "quant": wan_model_quant_family(low_id) if low_id else "",
        },
        "vae": {"name": Path(str(vae_id or "")).name, "generation": wan_vae_generation(vae_id)},
        "text_encoder": {"name": Path(str(text_encoder_id or "")).name, "kind": wan_text_encoder_kind(text_encoder_id)},
        "warnings": [],
        "lines": [],
    }
    warnings = []
    if dual:
        chk = wan_model_pair_compatibility(high_id, low_id)
        summary["pair_ok"] = chk.ok
        warnings.extend(chk.errors)
        warnings.extend(chk.warnings)
        # VAE generation should be 2.1 for the A14B dual pair.
        if summary["vae"]["generation"] == "2.2":
            warnings.append("A14B high/low expects the Wan 2.1 (16-ch) VAE; selected VAE looks 2.2 (48-ch).")
    else:
        summary["pair_ok"] = True
        if summary["vae"]["generation"] == "2.1":
            warnings.append("5B TI2V expects the Wan 2.2 (48-ch) VAE; selected VAE looks 2.1 (16-ch).")
    summary["warnings"] = warnings

    L = summary["lines"]
    L.append(f"Runtime: {runtime_mode or 'unknown'}  ({'dual high/low' if dual else 'single transformer'})")
    if dual:
        L.append(f"High: {summary['high']['name'] or '-'}  [{summary['high']['storage']}/{summary['high']['quant']}]")
        L.append(f"Low : {summary['low']['name'] or '-'}  [{summary['low']['storage']}/{summary['low']['quant']}]")
    else:
        L.append(f"Model: {summary['high']['name'] or summary['low']['name'] or '(5B base)'}")
    L.append(f"VAE: {summary['vae']['name'] or '(default)'}  -> Wan {summary['vae']['generation']}")
    L.append(f"Text encoder: {summary['text_encoder']['name'] or '(component base)'}  [{summary['text_encoder']['kind']}]")
    return summary


# --- Header-based identification (ground truth from the file, not the name) ---

@dataclass(frozen=True)
class WanHeaderInfo:
    ok: bool = False
    storage: str = ""        # gguf | safetensors | ...
    arch: str = ""           # e.g. "wan_2.2_14b_i2v" (safetensors) or "wan" (gguf)
    size_class: str = ""     # "5b" | "14b" | ""
    in_channels: int = 0     # 36 (14B I2V) | 48 (5B TI2V) | 52 (Fun-Control) | 32 (T2V) | 0
    needs_vae: str = ""      # "2.1" (16-ch) | "2.2" (48-ch) | ""
    quant: str = ""          # fp8 | q2..q8 | bf16 | fp16 | ""
    role: str = ""           # high | low | ""  (header-derived only)
    title: str = ""
    tensors: int = 0
    error: str = ""


def _wan_vae_for_in_channels(ic: int) -> str:
    if ic == 48:
        return "2.2"
    if ic in (32, 36, 52):
        return "2.1"
    return ""


def _wan_size_class(dim: int, n_blocks: int, tensors: int) -> str:
    if dim >= 5000 or n_blocks >= 40:
        return "14b"
    if (3000 <= dim < 5000) or n_blocks == 30:
        return "5b"
    if tensors >= 1500:
        return "14b"
    return ""


def _wan_role_from_text(s: str) -> str:
    s = str(s or "").lower()
    if "high" in s and "low" not in s:
        return "high"
    if "low" in s and "high" not in s:
        return "low"
    return ""


def wan_model_header_info(path) -> "WanHeaderInfo":
    """Read just the header/metadata of a Wan transformer file (cheap; no tensor load).

    safetensors: parse the JSON header (stdlib). gguf: read metadata + tensor shapes
    (lazy gguf import). Returns architecture, size class, in-channels (-> required VAE),
    quant tier, and high/low role when the header records it. ok=False on any problem.
    """
    import json as _json, struct as _struct, collections as _c
    p = Path(str(path or ""))
    storage = wan_model_storage_family(str(p))
    try:
        if storage == "safetensors":
            with open(p, "rb") as f:
                n = _struct.unpack("<Q", f.read(8))[0]
                hdr = _json.loads(f.read(n))
            meta = hdr.pop("__metadata__", {}) or {}
            dts = {v.get("dtype") for v in hdr.values() if isinstance(v, dict)}
            quant = ("fp8" if any(d and d.startswith("F8") for d in dts)
                     else "bf16" if "BF16" in dts else "fp16" if "F16" in dts else "")
            ic = dim = 0
            for k, v in hdr.items():
                if k.endswith("patch_embedding.weight") and isinstance(v, dict):
                    vals = [int(x) for x in (v.get("shape") or [])]
                    ic = next((x for x in vals if x in (32, 36, 48, 52)), 0)
                    dim = max(vals) if vals else 0
                    break
            nb = len({k.split("blocks.")[1].split(".")[0] for k in hdr if "blocks." in k})
            arch = str(meta.get("modelspec.architecture", ""))
            title = str(meta.get("modelspec.title", ""))
            return WanHeaderInfo(True, storage, arch, _wan_size_class(dim, nb, len(hdr)),
                                 ic, _wan_vae_for_in_channels(ic), quant,
                                 _wan_role_from_text(title) or _wan_role_from_text(arch),
                                 title, len(hdr))
        if storage == "gguf":
            import gguf
            r = gguf.GGUFReader(str(p))
            tensors = list(r.tensors)
            qc = _c.Counter(str(t.tensor_type).split(".")[-1] for t in tensors)
            quant = ""
            for tier in ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0", "Q4_0", "Q5_0"):
                if qc.get(tier):
                    quant = "q" + tier[1]
                    break
            ic = dim = 0
            for t in tensors:
                if str(t.name).endswith("patch_embedding.weight"):
                    vals = [int(x) for x in list(t.shape)]
                    ic = next((x for x in vals if x in (32, 36, 48)), 0)
                    dim = max(vals) if vals else 0
                    break
            arch = ""
            try:
                fld = r.fields.get("general.architecture")
                arch = bytes(fld.parts[fld.data[0]]).decode("utf-8", "replace") if fld else ""
            except Exception:
                arch = ""
            return WanHeaderInfo(True, storage, arch, _wan_size_class(dim, 0, len(tensors)),
                                 ic, _wan_vae_for_in_channels(ic), quant, "", "", len(tensors))
    except Exception as e:
        return WanHeaderInfo(False, storage, error=f"{type(e).__name__}: {e}")
    return WanHeaderInfo(False, storage, error="unsupported storage")


def wan_autopair(selected_id, candidates, *, want_role: str):
    """Best counterpart using HEADER ground truth first, filename as fallback.

    Header match requires: same storage, same size_class, same in_channels, same quant,
    and opposite role (role read from each file's header, else its filename). If headers
    are unavailable/ambiguous, falls back to the filename-based matcher.
    """
    sel = str(selected_id or "").strip()
    if not sel or want_role not in ("high", "low"):
        return None
    sh = wan_model_header_info(sel)
    if sh.ok and (sh.size_class or sh.in_channels or sh.quant):
        sel_role = sh.role or wan_model_stage_role(sel)
        sel_name = Path(sel).name
        best, best_score = None, 0
        for cand in candidates or ():
            cid = str(cand or "").strip()
            if not cid or Path(cid).name == sel_name:
                continue
            ch = wan_model_header_info(cid)
            cand_role = (ch.role if ch.ok and ch.role else wan_model_stage_role(cid))
            if cand_role != want_role:
                continue
            if ch.ok:
                if sh.storage and ch.storage and ch.storage != sh.storage:
                    continue
                if sh.size_class and ch.size_class and ch.size_class != sh.size_class:
                    continue
                if sh.in_channels and ch.in_channels and ch.in_channels != sh.in_channels:
                    continue
                score = 3
                if sh.quant and ch.quant and ch.quant == sh.quant:
                    score += 2
                if sh.arch and ch.arch and ch.arch == sh.arch:
                    score += 2
            else:
                if wan_model_storage_family(cid) != sh.storage:
                    continue
                score = 1
            if score > best_score:
                best, best_score = cid, score
        if best:
            return best
    # fallback: filename-only matcher
    return wan_autopair_counterpart(sel, candidates, want_role=want_role)


# --- Runtime-driven file filtering (runtime choice is the master filter) ---
# UX rule (Shawn): the selected runtime decides which files show in EVERY dropdown.
# Within a format you may mix high/low from different families AS LONG AS the quant tier
# matches. Storage comes free from the extension, so this works even with no readable header.

def wan_runtime_size_class(runtime_mode: str) -> str:
    rm = str(runtime_mode or "")
    if rm == "fast_5b":
        return "5b"
    if rm in ("native_high_low", "native_high_low_fp8_experimental"):
        return "14b"
    return ""


def wan_model_matches(model_id, *, storage=None, quant=None, size_class=None,
                      role=None, use_header=True) -> bool:
    """Does ``model_id`` satisfy the constraints? Storage/quant/role come from the filename
    (always available); header (when readable) upgrades storage/quant/size/role. Unknown
    fields never exclude — only a concrete mismatch does."""
    sid = str(model_id or "")
    if not sid:
        return False
    st = wan_model_storage_family(sid)
    q = wan_model_quant_family(sid)
    rl = wan_model_stage_role(sid)
    sc = ""
    if use_header:
        h = wan_model_header_info(sid)
        if h.ok:
            st = h.storage or st
            q = h.quant or q
            rl = h.role or rl
            sc = h.size_class
    if storage and st != "unknown" and st != storage:
        return False
    if quant and quant != "unknown" and q != "unknown" and q != quant:
        return False
    if size_class and sc and sc != size_class:
        return False
    if role and rl not in (role, "unknown"):
        return False
    return True


def _peer_storage_quant(peer_id):
    st = wan_model_storage_family(str(peer_id or ""))
    q = wan_model_quant_family(str(peer_id or ""))
    h = wan_model_header_info(str(peer_id or "")) if peer_id else None
    if h and h.ok:
        st = h.storage or st
        q = h.quant or q
    return st, q


def wan_selectable_transformers(candidates, *, runtime_mode, want_role=None, peer_id=None):
    """Subset of transformer files selectable for one dropdown given the runtime (size class)
    and, if the peer stage is already chosen, the same storage + quant as the peer."""
    sc = wan_runtime_size_class(runtime_mode)
    peer_st = peer_q = None
    if peer_id:
        peer_st, peer_q = _peer_storage_quant(peer_id)
    out = []
    for c in candidates or ():
        cid = str(c or "")
        if not cid:
            continue
        if peer_id and Path(cid).name == Path(str(peer_id)).name:
            continue
        if wan_model_matches(cid, storage=peer_st, quant=peer_q, size_class=sc, role=want_role):
            out.append(cid)
    return out


def wan_lora_matches(lora_id, *, size_class="") -> bool:
    """Best-effort LoRA compatibility by filename (A14B/14B vs 5B)."""
    n = Path(str(lora_id or "")).name.lower()
    if not size_class:
        return True
    has14 = "a14b" in n or "14b" in n
    has5 = "5b" in n or "ti2v" in n
    if size_class == "14b":
        return not (has5 and not has14)
    if size_class == "5b":
        return not (has14 and not has5)
    return True


def wan_selectable_loras(candidates, *, runtime_mode):
    sc = wan_runtime_size_class(runtime_mode)
    return [str(c) for c in (candidates or ()) if c and wan_lora_matches(c, size_class=sc)]
