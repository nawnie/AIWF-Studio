// Path, file and formatting helpers (see Paths.h).
#include "pch.h"
#include "Services/Paths.h"
#include <shlobj.h>

namespace aiwf::paths
{
    std::filesystem::path ExeDir()
    {
        // this loop grows the buffer until the full module path fits (long paths are allowed)
        std::wstring buffer(MAX_PATH, L'\0');
        for (;;)
        {
            DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
            if (length < buffer.size())
            {
                buffer.resize(length);
                break;
            }
            buffer.resize(buffer.size() * 2);
        }
        return std::filesystem::path(buffer).parent_path();
    }

    // ---- the Studio code folder ---------------------------------------------------------------------------
    namespace
    {
        std::mutex g_rootLock;
        std::filesystem::path g_rootOverride;   // chosen in setup; saved in settings.json by AppState
    }

    bool IsStudioRoot(std::filesystem::path const& folder)
    {
        std::error_code ignored;
        return !folder.empty() && std::filesystem::exists(folder / L"aiwf" / L"engine_api.py", ignored);
    }

    void SetStudioRootOverride(std::filesystem::path const& folder)
    {
        std::lock_guard guard(g_rootLock);
        g_rootOverride = folder;
    }

    std::filesystem::path StudioRoot()
    {
        // 1. an explicit environment variable wins (developers, scripted installs)
        wchar_t configured[32768]{};
        DWORD configuredLength = GetEnvironmentVariableW(L"AIWF_STUDIO_ROOT", configured, static_cast<DWORD>(std::size(configured)));
        if (configuredLength > 0 && configuredLength < std::size(configured))
        {
            std::error_code error;
            auto path = std::filesystem::weakly_canonical(configured, error);
            if (!error && IsStudioRoot(path)) return path;
        }

        // 2. the folder the person picked in setup (an app installed outside the code folder)
        {
            std::lock_guard guard(g_rootLock);
            if (IsStudioRoot(g_rootOverride)) return g_rootOverride;
        }

        // 3. the app sits inside the code folder (native\bin\Release or native\installed)
        auto current = ExeDir();
        while (!current.empty())
        {
            if (IsStudioRoot(current)) return current;
            auto parent = current.parent_path();
            if (parent == current) break;
            current = std::move(parent);
        }
        return ExeDir();
    }

    std::filesystem::path DataDir()
    {
        static std::filesystem::path dir = []
        {
            wil::unique_cotaskmem_string local;
            std::filesystem::path base;
            if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_LocalAppData, KF_FLAG_CREATE, nullptr, &local)))
            {
                base = local.get();
            }
            else
            {
                base = std::filesystem::temp_directory_path();
            }
            auto path = base / L"AIWF Studio";
            std::error_code ignored;
            std::filesystem::create_directories(path, ignored);
            return path;
        }();
        return dir;
    }

    std::filesystem::path LogsDir()
    {
        auto path = DataDir() / L"logs";
        std::error_code ignored;
        std::filesystem::create_directories(path, ignored);
        return path;
    }

    std::wstring Expand(std::wstring const& text)
    {
        std::wstring input = text;
        constexpr std::wstring_view rootToken = L"%AIWF_STUDIO_ROOT%";
        size_t position = 0;
        auto root = StudioRoot().wstring();
        while ((position = input.find(rootToken, position)) != std::wstring::npos)
        {
            input.replace(position, rootToken.size(), root);
            position += root.size();
        }
        if (input.find(L'%') == std::wstring::npos) return input;
        DWORD needed = ExpandEnvironmentStringsW(input.c_str(), nullptr, 0);
        if (needed == 0) return input;
        std::wstring out(needed, L'\0');
        ExpandEnvironmentStringsW(input.c_str(), out.data(), needed);
        out.resize(needed - 1);   // drop the terminating null the API counts
        return out;
    }

    std::wstring Timestamp()
    {
        SYSTEMTIME now{};
        GetLocalTime(&now);
        wchar_t buffer[32]{};
        swprintf_s(buffer, L"%04u%02u%02u-%02u%02u%02u", now.wYear, now.wMonth, now.wDay, now.wHour, now.wMinute, now.wSecond);
        return buffer;
    }

    std::optional<std::string> ReadUtf8(std::filesystem::path const& path)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file) return std::nullopt;
        std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEF && static_cast<unsigned char>(text[1]) == 0xBB &&
            static_cast<unsigned char>(text[2]) == 0xBF)
        {
            text.erase(0, 3);
        }
        return text;
    }

    bool WriteUtf8Atomic(std::filesystem::path const& path, std::string_view text)
    {
        // write a sibling temp file, then swap it in, so a crash never leaves half a settings file
        auto temp = path;
        temp += L".tmp";
        {
            std::ofstream file(temp, std::ios::binary | std::ios::trunc);
            if (!file) return false;
            file.write(text.data(), static_cast<std::streamsize>(text.size()));
            if (!file) return false;
        }
        return MoveFileExW(temp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != FALSE;
    }

    std::wstring FirstKeyLine(std::filesystem::path const& path)
    {
        auto text = ReadUtf8(path);
        if (!text) return {};
        std::istringstream lines(*text);
        std::string line;
        // this loop skips blank lines and # comments, like the engines' own key readers
        while (std::getline(lines, line))
        {
            auto begin = line.find_first_not_of(" \t\r");
            if (begin == std::string::npos || line[begin] == '#') continue;
            auto end = line.find_last_not_of(" \t\r");
            return std::wstring(winrt::to_hstring(line.substr(begin, end - begin + 1)));
        }
        return {};
    }

    std::wstring FormatBytes(uint64_t bytes)
    {
        wchar_t buffer[32]{};
        double value = static_cast<double>(bytes);
        if (value >= 1024.0 * 1024 * 1024)
            swprintf_s(buffer, L"%.1f GB", value / (1024.0 * 1024 * 1024));
        else if (value >= 1024.0 * 1024)
            swprintf_s(buffer, L"%.0f MB", value / (1024.0 * 1024));
        else
            swprintf_s(buffer, L"%.0f KB", value / 1024.0);
        return buffer;
    }
}
