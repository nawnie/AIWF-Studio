// Where the app keeps things, plus small file and text helpers shared by the services.
//
//   AESStudio.exe folder       shipped files: engines.json, the Windows App SDK runtime, resources.pri
//   %LOCALAPPDATA%\Programs\AES Studio Data  per-user state: settings.json, an optional engines.json override, and
//                               the logs folder (one log per engine start)
#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace aiwf::paths
{
    struct NativeBuildInfo
    {
        bool valid = false;
        uint32_t protocolVersion = 0;
        std::wstring apiSourceFingerprint;
        std::wstring error;
    };

    struct InstalledStudioRootBinding
    {
        std::filesystem::path root;
        uint32_t protocolVersion = 0;
        std::wstring apiSourceFingerprint;
    };

    std::filesystem::path ExeDir();
    // The AES Studio code folder (the one containing aiwf/engine_api.py). Looked up in this order:
    // the AES_STUDIO_ROOT environment variable, the per-install studio-root.json binding, the
    // folder chosen in setup (for unbound/dev runs), then the nearest parent of the exe.
    std::filesystem::path StudioRoot();
    void SetStudioRootOverride(std::filesystem::path const& folder);   // empty clears it
    bool IsStudioRoot(std::filesystem::path const& folder);            // has aiwf\engine_api.py
    NativeBuildInfo ReadNativeBuildInfo();                             // adjacent native-build-info.json; missing is invalid
    std::optional<InstalledStudioRootBinding> ReadInstalledStudioRootBinding();
    bool ValidateStudioRootSelection(std::filesystem::path const& folder, std::wstring& problem);

    inline bool IsLowerSha256(std::wstring_view value) noexcept
    {
        if (value.size() != 64) return false;
        return std::all_of(value.begin(), value.end(), [](wchar_t character)
        {
            return (character >= L'0' && character <= L'9') || (character >= L'a' && character <= L'f');
        });
    }

    inline std::filesystem::path SelectStudioRootCandidate(
        std::optional<std::filesystem::path> processOverride,
        std::optional<std::filesystem::path> installedBinding,
        std::optional<std::filesystem::path> savedSetting,
        std::filesystem::path fallback)
    {
        if (processOverride && !processOverride->empty()) return *processOverride;
        if (installedBinding && !installedBinding->empty()) return *installedBinding;
        if (savedSetting && !savedSetting->empty()) return *savedSetting;
        return fallback;
    }

    std::filesystem::path DataDir();   // created on first use; AES_STUDIO_DATA_DIR may select an isolated profile root
    std::filesystem::path LogsDir();   // DataDir\logs, created on first use

    // expands %VARIABLE% references (engines.json uses %LOCALAPPDATA%, %USERPROFILE%, %ComSpec%)
    std::wstring Expand(std::wstring const& text);

    // local time as 20261007-043512, for log file names
    std::wstring Timestamp();

    // whole-file UTF-8 read/write (a UTF-8 byte order mark is skipped on read); write is atomic
    std::optional<std::string> ReadUtf8(std::filesystem::path const& path);
    bool WriteUtf8Atomic(std::filesystem::path const& path, std::string_view text);

    // the first non-empty line that is not a # comment: the format of every local key file
    std::wstring FirstKeyLine(std::filesystem::path const& path);

    // "12.3 GB" style sizes for the UI
    std::wstring FormatBytes(uint64_t bytes);
}
