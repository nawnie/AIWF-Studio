"""Read AES Studio provider keys from the current Windows account only."""
from __future__ import annotations
import ctypes
import os
from ctypes import wintypes

class _Credential(ctypes.Structure):
    _fields_ = [("Flags", wintypes.DWORD), ("Type", wintypes.DWORD),
        ("TargetName", wintypes.LPWSTR), ("Comment", wintypes.LPWSTR),
        ("LastWritten", wintypes.FILETIME), ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)), ("Persist", wintypes.DWORD),
        ("AttributeCount", wintypes.DWORD), ("Attributes", ctypes.c_void_p),
        ("TargetAlias", wintypes.LPWSTR), ("UserName", wintypes.LPWSTR)]

_TARGETS = {"huggingface": "AESStudio.HuggingFace", "civitai": "AESStudio.Civitai"}

def read_provider_key(provider: str) -> str:
    target = _TARGETS.get(provider)
    if os.name != "nt" or not target:
        return ""
    advapi = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
    read = advapi.CredReadW
    read.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(ctypes.POINTER(_Credential))]
    read.restype = wintypes.BOOL
    free = advapi.CredFree
    free.argtypes = [ctypes.c_void_p]
    item = ctypes.POINTER(_Credential)()
    if not read(target, 1, 0, ctypes.byref(item)):
        return ""
    try:
        size = item.contents.CredentialBlobSize
        if not size or size > 2560 or size % 2:
            return ""
        return ctypes.string_at(item.contents.CredentialBlob, size).decode("utf-16-le")
    finally:
        free(item)
