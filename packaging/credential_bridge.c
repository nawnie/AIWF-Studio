#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <wincred.h>
#include <wchar.h>

int __stdcall SaveKey(const wchar_t *provider, const wchar_t *secret) {
  const wchar_t *target = 0;
  if (wcscmp(provider, L"huggingface") == 0) target = L"AESStudio.HuggingFace";
  if (wcscmp(provider, L"civitai") == 0) target = L"AESStudio.Civitai";
  if (!target || !secret) return 0;
  size_t len = wcslen(secret);
  if (!len || len * sizeof(wchar_t) > CRED_MAX_CREDENTIAL_BLOB_SIZE) return 0;
  CREDENTIALW cred = {0};
  cred.Type = CRED_TYPE_GENERIC;
  cred.TargetName = (LPWSTR)target;
  cred.CredentialBlob = (LPBYTE)secret;
  cred.CredentialBlobSize = (DWORD)(len * sizeof(wchar_t));
  cred.Persist = CRED_PERSIST_LOCAL_MACHINE;
  cred.UserName = L"AES Studio";
  return CredWriteW(&cred, 0) ? 1 : 0;
}
