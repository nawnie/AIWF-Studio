// Where the app keeps things, plus small file and text helpers shared by the services.
//
//   AIWFStudio.exe folder       shipped files: engines.json, the Windows App SDK runtime, resources.pri
//   %LOCALAPPDATA%\AIWF Studio  per-user state: settings.json, an optional engines.json override, and
//                               the logs folder (one log per engine start)
#pragma once

namespace aiwf::paths
{
    std::filesystem::path ExeDir();
    // The AIWF Studio code folder (the one containing aiwf/engine_api.py). Looked up in this order:
    // the AIWF_STUDIO_ROOT environment variable, the folder chosen in setup (SetStudioRootOverride),
    // then the nearest parent of the exe. Falls back to the exe folder when none qualifies.
    std::filesystem::path StudioRoot();
    void SetStudioRootOverride(std::filesystem::path const& folder);   // empty clears it
    bool IsStudioRoot(std::filesystem::path const& folder);            // has aiwf\engine_api.py
    std::filesystem::path DataDir();   // created on first use
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
