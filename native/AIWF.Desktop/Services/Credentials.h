// AES Studio provider credentials use Windows Credential Manager, per Windows account.
#pragma once
#include <windows.h>
#include <wincred.h>
#include <string>
#include <string_view>
namespace aiwf::credentials {
inline wchar_t const* Target(std::wstring_view provider) noexcept {
    if (provider == L"Hugging Face") return L"AESStudio.HuggingFace";
    if (provider == L"Civitai") return L"AESStudio.Civitai";
    return nullptr;
}
inline bool Has(std::wstring_view provider) noexcept {
    auto target = Target(provider); PCREDENTIALW cred = nullptr;
    if (!target || !CredReadW(target, CRED_TYPE_GENERIC, 0, &cred)) return false;
    CredFree(cred); return true;
}
inline bool Write(std::wstring_view provider, std::wstring const& secret) noexcept {
    auto target = Target(provider);
    if (!target || secret.empty() || secret.size() * sizeof(wchar_t) > CRED_MAX_CREDENTIAL_BLOB_SIZE) return false;
    CREDENTIALW cred{}; cred.Type = CRED_TYPE_GENERIC; cred.TargetName = const_cast<LPWSTR>(target);
    cred.CredentialBlobSize = static_cast<DWORD>(secret.size() * sizeof(wchar_t));
    cred.CredentialBlob = reinterpret_cast<LPBYTE>(const_cast<wchar_t*>(secret.data()));
    cred.Persist = CRED_PERSIST_LOCAL_MACHINE; cred.UserName = const_cast<LPWSTR>(L"AES Studio");
    return CredWriteW(&cred, 0) != FALSE;
}
inline bool Remove(std::wstring_view provider) noexcept {
    auto target = Target(provider); if (!target) return false;
    return CredDeleteW(target, CRED_TYPE_GENERIC, 0) || GetLastError() == ERROR_NOT_FOUND;
}
}
