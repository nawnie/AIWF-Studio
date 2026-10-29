"""Opt-in image-model downloads for the AES Studio native preview.

No user-provided URL is fetched. The UI supplies an HF repository/file or a
numeric Civitai model-version ID; only known HTTPS provider hosts may redirect.
Credentials never enter command lines, logs, saved settings, or job responses.
"""
from __future__ import annotations
import hashlib
import os
import re
import shutil
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from uuid import uuid4

from aiwf.credentials import read_provider_key
from aiwf.infrastructure.diffusers.model_arch import gguf_has_shallow_header
from aiwf.infrastructure.safetensors_metadata import safetensors_file_is_structurally_valid

MAX_DOWNLOAD = 50 * 1024**3
RESERVE_BYTES = 512 * 1024**2
SDXL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_FILE = "sd_xl_base_1.0.safetensors"
SDXL_SHA256 = "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b"
MODEL_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".gguf"}
LEGACY_EXTENSIONS = {".ckpt", ".pt", ".pth"}
_SAFE_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z", re.ASCII)

class DownloadProblem(ValueError):
    pass

class DownloadCancelled(Exception):
    pass

def _allowed_host(source: str, hostname: str) -> bool:
    host = hostname.lower().rstrip(".")
    def under(domain: str) -> bool:
        return host == domain or host.endswith("." + domain)
    if source == "huggingface":
        return under("huggingface.co") or under("hf.co")
    if source == "civitai":
        return under("civitai.com") or under("civitai.green") or under("r2.cloudflarestorage.com")
    return False

def _trusted_url(source: str, url: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    return parsed.scheme.lower() == "https" and parsed.username is None and parsed.password is None and (parsed.port in (None, 443)) and _allowed_host(source, parsed.hostname or "")

class _ProviderRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, source: str):
        self.source = source
        super().__init__()
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _trusted_url(self.source, newurl):
            raise DownloadProblem("The provider redirected to an unsupported host. No file was installed.")
        result = super().redirect_request(req, fp, code, msg, headers, newurl)
        if result is not None and urllib.parse.urlsplit(req.full_url).netloc.lower() != urllib.parse.urlsplit(newurl).netloc.lower():
            result.remove_header("Authorization")
        return result

def _request(source: str, reference: str, filename: str) -> tuple[str, str, str | None]:
    reference, filename = reference.strip(), filename.strip()
    if source == "huggingface":
        repo_parts = reference.split("/")
        file_parts = filename.split("/")
        if len(repo_parts) != 2 or not all(_SAFE_PART.fullmatch(p) and p not in {".", ".."} for p in repo_parts):
            raise DownloadProblem("Enter a Hugging Face repository as owner/model.")
        if not file_parts or len(file_parts) > 8 or not all(_SAFE_PART.fullmatch(p) and p not in {".", ".."} for p in file_parts):
            raise DownloadProblem("Enter the model filename or repository file path.")
        name = file_parts[-1]
        url = "https://huggingface.co/" + "/".join(repo_parts) + "/resolve/main/" + "/".join(file_parts)
        expected = SDXL_SHA256 if reference == SDXL_REPO and filename == SDXL_FILE else None
    elif source == "civitai":
        if not re.fullmatch(r"[0-9]{1,12}", reference, re.ASCII) or int(reference) < 1:
            raise DownloadProblem("Enter the numeric Civitai model version ID.")
        if not _SAFE_PART.fullmatch(filename) or filename in {".", ".."}:
            raise DownloadProblem("Enter a simple model filename.")
        name = filename
        url = "https://civitai.com/api/download/models/" + reference
        expected = None
    else:
        raise DownloadProblem("Choose Hugging Face or Civitai.")
    if Path(name).suffix.casefold() not in MODEL_EXTENSIONS:
        raise DownloadProblem("Choose a .safetensors, .ckpt, .pt, .pth or .gguf model file.")
    return url, name, expected

class ModelDownloads:
    def __init__(self, flags):
        self.flags = flags
        self.lock = threading.Lock()
        self.jobs = {}
        self.cancel = {}

    def start(self, source: str, reference: str, filename: str) -> dict:
        url, name, expected = _request(source, reference, filename)
        root = Path(self.flags.resolved_ckpt_dir())
        destination = root / name
        if destination.exists():
            raise DownloadProblem("A file with that name already exists in Image checkpoints. It was not changed.")
        with self.lock:
            if any(job["state"] == "running" for job in self.jobs.values()):
                raise DownloadProblem("A checkpoint download is already running.")
            ident = uuid4().hex
            job = {"id": ident, "state": "running", "filename": name, "bytes_received": 0, "bytes_total": 0, "message": "Connecting to the provider..."}
            self.jobs[ident] = job
            event = self.cancel[ident] = threading.Event()
        threading.Thread(target=self._run, args=(ident, source, url, destination, expected, event), daemon=True).start()
        return dict(job)

    def active(self) -> dict | None:
        with self.lock:
            return next((dict(job) for job in self.jobs.values() if job["state"] == "running"), None)

    def status(self, ident: str) -> dict | None:
        with self.lock:
            job = self.jobs.get(ident)
            return dict(job) if job else None

    def stop(self, ident: str) -> bool:
        with self.lock:
            event = self.cancel.get(ident)
            if not event or self.jobs[ident]["state"] != "running": return False
            event.set()
            self.jobs[ident]["message"] = "Cancelling..."
            return True

    def _update(self, ident: str, **values):
        with self.lock: self.jobs[ident].update(values)

    def _run(self, ident, source, url, destination, expected, event):
        temporary = None
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists(): raise DownloadProblem("A file with that name already exists; it was not changed.")
            if not _trusted_url(source, url): raise DownloadProblem("Unsupported provider URL.")
            token = read_provider_key(source)
            headers = {"User-Agent": "AES-Studio/0.1"}
            if token: headers["Authorization"] = "Bearer " + token
            request = urllib.request.Request(url, headers=headers)
            opener = urllib.request.build_opener(_ProviderRedirect(source))
            with opener.open(request, timeout=30) as response:
                length_text = response.headers.get("Content-Length", "0")
                try: total = int(length_text)
                except (TypeError, ValueError): total = 0
                if total < 0 or total > MAX_DOWNLOAD: raise DownloadProblem("The download exceeds the 50 GiB preview limit.")
                if total and shutil.disk_usage(destination.parent).free < total + RESERVE_BYTES:
                    raise DownloadProblem("There is not enough free disk space for this download.")
                fd, name = tempfile.mkstemp(prefix=".aes-download-", suffix=".partial", dir=destination.parent)
                temporary = Path(name)
                digest = hashlib.sha256(); done = 0; checked = 0
                self._update(ident, bytes_total=total, message="Downloading...")
                with os.fdopen(fd, "wb") as output:
                    while True:
                        if event.is_set(): raise DownloadCancelled()
                        block = response.read(1024 * 1024)
                        if not block: break
                        done += len(block)
                        if done > MAX_DOWNLOAD: raise DownloadProblem("The download exceeds the 50 GiB preview limit.")
                        if done - checked >= 64 * 1024**2:
                            if shutil.disk_usage(destination.parent).free < RESERVE_BYTES:
                                raise DownloadProblem("The download stopped to preserve free disk space.")
                            checked = done
                        output.write(block); digest.update(block)
                        self._update(ident, bytes_received=done)
                if event.is_set(): raise DownloadCancelled()
                if total and done != total: raise DownloadProblem("The download ended before all advertised bytes arrived.")
                extension = destination.suffix.casefold()
                if not done:
                    raise DownloadProblem("The provider returned an empty file; it was not installed.")
                if extension == ".safetensors" and not safetensors_file_is_structurally_valid(temporary):
                    raise DownloadProblem("The downloaded file is not a structurally valid safetensors file.")
                if extension == ".gguf" and not gguf_has_shallow_header(temporary):
                    raise DownloadProblem("The downloaded file does not have a valid GGUF header; it was not installed.")
                if expected and digest.hexdigest() != expected:
                    raise DownloadProblem("The official SDXL file did not match its expected SHA-256; it was not installed.")
                # Windows rename fails rather than replacing an existing destination.
                if os.name == "nt": os.rename(temporary, destination)
                else:
                    os.link(temporary, destination)
                    temporary.unlink()
                temporary = None
                message = "Saved to Image checkpoints. Refresh backend readiness to see whether this model is compatible."
                if extension in LEGACY_EXTENSIONS:
                    message += " Legacy PyTorch content was not validated and can execute code when loaded; use trusted models only."
                elif extension == ".gguf":
                    message += " Only the GGUF header was checked; compatibility and integrity are not verified."
                self._update(ident, state="completed", message=message)
        except DownloadCancelled:
            self._update(ident, state="cancelled", message="Download cancelled. A partial file may remain if Windows could not remove it.")
        except urllib.error.HTTPError as exc:
            self._update(ident, state="failed", message=f"The provider returned HTTP {exc.code}. Check access, license and your key.")
        except DownloadProblem as exc:
            self._update(ident, state="failed", message=str(exc))
        except FileExistsError:
            self._update(ident, state="failed", message="A file with that name already exists; it was not changed.")
        except Exception as exc:
            self._update(ident, state="failed", message=f"The download could not finish ({type(exc).__name__}). Check your connection and folder permissions.")
        finally:
            if temporary is not None:
                try: temporary.unlink(missing_ok=True)
                except OSError: pass
