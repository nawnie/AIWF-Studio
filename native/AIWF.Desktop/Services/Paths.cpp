// Path, file and formatting helpers (see Paths.h).
#include "pch.h"
#include "Services/Paths.h"
#include "Services/DataDirOverride.h"
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

        std::optional<std::filesystem::path> ReadProcessStudioRootOverride()
        {
            SetLastError(ERROR_SUCCESS);
            DWORD required = GetEnvironmentVariableW(L"AES_STUDIO_ROOT", nullptr, 0);
            if (required == 0)
            {
                DWORD error = GetLastError();
                if (error == ERROR_ENVVAR_NOT_FOUND) return std::nullopt;
                throw winrt::hresult_error(E_INVALIDARG,
                    L"AES_STUDIO_ROOT is set but empty or unreadable. Set it to a checkout containing aiwf\\engine_api.py or remove the variable.");
            }

            std::wstring value(required, L'\0');
            DWORD copied = GetEnvironmentVariableW(L"AES_STUDIO_ROOT", value.data(), required);
            if (copied == 0 || copied >= required)
            {
                throw winrt::hresult_error(E_INVALIDARG,
                    L"AES_STUDIO_ROOT changed while it was being read. Set it to a checkout containing aiwf\\engine_api.py or remove the variable.");
            }
            value.resize(copied);
            std::filesystem::path root(value);
            if (!root.is_absolute())
            {
                throw winrt::hresult_error(E_INVALIDARG, L"AES_STUDIO_ROOT must be an absolute path.");
            }
            std::error_code error;
            root = std::filesystem::weakly_canonical(root, error);
            if (error || !IsStudioRoot(root))
            {
                throw winrt::hresult_error(E_INVALIDARG,
                    L"AES_STUDIO_ROOT does not contain aiwf\\engine_api.py: " + root.wstring());
            }
            return root;
        }

        bool SamePath(std::filesystem::path const& left, std::filesystem::path const& right)
        {
            std::error_code leftError, rightError;
            auto leftFull = std::filesystem::weakly_canonical(left, leftError);
            auto rightFull = std::filesystem::weakly_canonical(right, rightError);
            if (leftError || rightError) return false;
            auto leftText = leftFull.lexically_normal().wstring();
            auto rightText = rightFull.lexically_normal().wstring();
            return _wcsicmp(leftText.c_str(), rightText.c_str()) == 0;
        }

        bool ProbeDataDirWritable(std::filesystem::path const& folder, DWORD& error)
        {
            static std::atomic_uint64_t sequence{ 0 };
            for (int attempt = 0; attempt < 4; ++attempt)
            {
                auto name = L".aiwf-write-probe-" + std::to_wstring(GetCurrentProcessId()) + L"-" +
                            std::to_wstring(GetTickCount64()) + L"-" + std::to_wstring(++sequence) + L".tmp";
                auto probePath = folder / name;
                wil::unique_hfile probe(CreateFileW(probePath.c_str(), GENERIC_WRITE | DELETE,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, CREATE_NEW,
                    FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr));
                if (!probe)
                {
                    error = GetLastError();
                    if (error == ERROR_FILE_EXISTS || error == ERROR_ALREADY_EXISTS) continue;
                    return false;
                }

                char marker = 'x';
                DWORD written = 0;
                BOOL writeSucceeded = WriteFile(probe.get(), &marker, sizeof(marker), &written, nullptr);
                if (!writeSucceeded || written != sizeof(marker))
                {
                    error = writeSucceeded ? ERROR_WRITE_FAULT : GetLastError();
                    probe.reset();
                    DeleteFileW(probePath.c_str());
                    return false;
                }
                if (!FlushFileBuffers(probe.get()))
                {
                    error = GetLastError();
                    probe.reset();
                    DeleteFileW(probePath.c_str());
                    return false;
                }
                probe.reset();
                if (!DeleteFileW(probePath.c_str()) && GetLastError() != ERROR_FILE_NOT_FOUND)
                {
                    error = GetLastError();
                    return false;
                }
                error = ERROR_SUCCESS;
                return true;
            }
            error = ERROR_FILE_EXISTS;
            return false;
        }
    }

    bool IsStudioRoot(std::filesystem::path const& folder)
    {
        std::error_code ignored;
        return !folder.empty() && std::filesystem::exists(folder / L"aiwf" / L"engine_api.py", ignored);
    }

    NativeBuildInfo ReadNativeBuildInfo()
    {
        NativeBuildInfo info;
        auto path = ExeDir() / L"native-build-info.json";
        auto fail = [&](std::wstring const& reason)
        {
            info.error = L"The native engine API compatibility manifest is invalid at " + path.wstring() + L": " + reason +
                         L" Reinstall AES Studio using its installer. If files remain after uninstalling, choose a new empty installation folder; your separate data folder is preserved.";
            return info;
        };

        DWORD attributes = GetFileAttributesW(path.c_str());
        if (attributes == INVALID_FILE_ATTRIBUTES)
        {
            return fail(L"the file is missing or cannot be opened.");
        }
        if ((attributes & FILE_ATTRIBUTE_DIRECTORY) || (attributes & FILE_ATTRIBUTE_REPARSE_POINT))
        {
            return fail(L"the file must be a regular file beside AESStudio.exe.");
        }
        auto text = ReadUtf8(path);
        if (!text) return fail(L"the file is unreadable.");
        winrt::Windows::Data::Json::JsonObject root{ nullptr };
        if (!winrt::Windows::Data::Json::JsonObject::TryParse(winrt::to_hstring(*text), root))
        {
            return fail(L"the file is not valid JSON.");
        }
        auto schema = root.TryLookup(L"schema");
        if (!schema || schema.ValueType() != winrt::Windows::Data::Json::JsonValueType::String ||
            schema.GetString() != L"aiwf-native-build-info-v1")
        {
            return fail(L"schema must be aiwf-native-build-info-v1.");
        }
        auto protocol = root.TryLookup(L"protocol_version");
        if (!protocol || protocol.ValueType() != winrt::Windows::Data::Json::JsonValueType::Number || protocol.GetNumber() != 2.0)
        {
            return fail(L"protocol_version must be the supported integer 2.");
        }
        auto files = root.TryLookup(L"api_source_files");
        if (!files || files.ValueType() != winrt::Windows::Data::Json::JsonValueType::Object || files.GetObject().Size() == 0)
        {
            return fail(L"api_source_files must be a non-empty object.");
        }
        for (auto const& entry : files.GetObject())
        {
            if (entry.Value().ValueType() != winrt::Windows::Data::Json::JsonValueType::String ||
                !IsLowerSha256(entry.Value().GetString()))
            {
                return fail(L"every api_source_files entry must contain a lowercase SHA-256 digest.");
            }
        }
        auto fingerprint = root.TryLookup(L"api_source_fingerprint");
        if (!fingerprint || fingerprint.ValueType() != winrt::Windows::Data::Json::JsonValueType::String ||
            !IsLowerSha256(fingerprint.GetString()))
        {
            return fail(L"api_source_fingerprint must be a lowercase SHA-256 digest.");
        }
        auto exeHash = root.TryLookup(L"exe_sha256");
        if (!exeHash || exeHash.ValueType() != winrt::Windows::Data::Json::JsonValueType::String ||
            !IsLowerSha256(exeHash.GetString()))
        {
            return fail(L"exe_sha256 must be a lowercase SHA-256 digest.");
        }
        info.protocolVersion = 2;
        info.apiSourceFingerprint = fingerprint.GetString();
        info.valid = true;
        return info;
    }

    std::optional<InstalledStudioRootBinding> ReadInstalledStudioRootBinding()
    {
        auto path = ExeDir() / L"studio-root.json";
        DWORD attributes = GetFileAttributesW(path.c_str());
        if (attributes == INVALID_FILE_ATTRIBUTES)
        {
            DWORD error = GetLastError();
            if (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND) return std::nullopt;
            throw winrt::hresult_error(HRESULT_FROM_WIN32(error),
                L"The per-install Studio root binding could not be read: " + path.wstring());
        }
        if ((attributes & FILE_ATTRIBUTE_DIRECTORY) || (attributes & FILE_ATTRIBUTE_REPARSE_POINT))
        {
            throw winrt::hresult_error(E_INVALIDARG,
                L"The per-install Studio root binding must be a regular file beside AESStudio.exe: " + path.wstring());
        }
        auto text = ReadUtf8(path);
        if (!text)
        {
            throw winrt::hresult_error(E_ACCESSDENIED,
                L"The per-install Studio root binding is unreadable: " + path.wstring());
        }
        winrt::Windows::Data::Json::JsonObject root{ nullptr };
        if (!winrt::Windows::Data::Json::JsonObject::TryParse(winrt::to_hstring(*text), root))
        {
            throw winrt::hresult_error(E_INVALIDARG,
                L"The per-install Studio root binding is not valid JSON: " + path.wstring());
        }
        auto schema = root.TryLookup(L"schema");
        auto rootValue = root.TryLookup(L"studio_root");
        auto protocol = root.TryLookup(L"protocol_version");
        auto fingerprint = root.TryLookup(L"api_source_fingerprint");
        if (!schema || schema.ValueType() != winrt::Windows::Data::Json::JsonValueType::String || schema.GetString() != L"aiwf-studio-root-v1" ||
            !rootValue || rootValue.ValueType() != winrt::Windows::Data::Json::JsonValueType::String || rootValue.GetString().empty() ||
            !protocol || protocol.ValueType() != winrt::Windows::Data::Json::JsonValueType::Number || protocol.GetNumber() != 2.0 ||
            !fingerprint || fingerprint.ValueType() != winrt::Windows::Data::Json::JsonValueType::String || !IsLowerSha256(fingerprint.GetString()))
        {
            throw winrt::hresult_error(E_INVALIDARG,
                L"The per-install Studio root binding is missing required schema, root, protocol or fingerprint fields: " + path.wstring());
        }

        std::filesystem::path rootPath(std::wstring(rootValue.GetString()));
        if (!rootPath.is_absolute())
        {
            throw winrt::hresult_error(E_INVALIDARG, L"The per-install Studio root must be an absolute path: " + rootPath.wstring());
        }
        std::error_code error;
        rootPath = std::filesystem::weakly_canonical(rootPath, error);
        if (error || !IsStudioRoot(rootPath))
        {
            throw winrt::hresult_error(E_INVALIDARG,
                L"The per-install Studio root does not contain aiwf\\engine_api.py: " + rootPath.wstring());
        }

        auto build = ReadNativeBuildInfo();
        if (!build.valid || build.protocolVersion != protocol.GetNumber() ||
            build.apiSourceFingerprint != fingerprint.GetString())
        {
            throw winrt::hresult_error(E_INVALIDARG,
                L"The per-install Studio root binding does not match native-build-info.json. Reinstall or rebuild this app with the selected Studio root.");
        }
        return InstalledStudioRootBinding{ rootPath, 2, std::wstring(fingerprint.GetString()) };
    }

    bool ValidateStudioRootSelection(std::filesystem::path const& folder, std::wstring& problem)
    {
        std::error_code error;
        auto selected = std::filesystem::weakly_canonical(folder, error);
        if (error || !selected.is_absolute() || !IsStudioRoot(selected))
        {
            problem = L"Choose an absolute folder containing aiwf\\engine_api.py.";
            return false;
        }
        try
        {
            auto processOverride = ReadProcessStudioRootOverride();
            if (processOverride)
            {
                if (!SamePath(selected, *processOverride))
                {
                    problem = L"AES_STUDIO_ROOT currently selects " + processOverride->wstring() +
                             L". Remove or change that process variable before selecting a different folder.";
                    return false;
                }
                problem.clear();
                return true;
            }
            if (auto binding = ReadInstalledStudioRootBinding())
            {
                if (!SamePath(selected, binding->root))
                {
                    problem = L"This installed copy is bound to " + binding->root.wstring() +
                             L". Rebuild or reinstall AES Studio with the folder you want to use.";
                    return false;
                }
            }
        }
        catch (winrt::hresult_error const& exception)
        {
            problem = std::wstring(exception.message());
            return false;
        }
        problem.clear();
        return true;
    }

    void SetStudioRootOverride(std::filesystem::path const& folder)
    {
        std::lock_guard guard(g_rootLock);
        g_rootOverride = folder;
    }

    std::filesystem::path StudioRoot()
    {
        auto processOverride = ReadProcessStudioRootOverride();
        std::optional<std::filesystem::path> binding;
        if (!processOverride)
        {
            if (auto installed = ReadInstalledStudioRootBinding()) binding = installed->root;
        }
        std::optional<std::filesystem::path> savedSetting;
        {
            std::lock_guard guard(g_rootLock);
            if (IsStudioRoot(g_rootOverride)) savedSetting = g_rootOverride;
        }

        // An uninstalled developer build keeps the nearest-parent lookup when no process or install binding exists.
        auto current = ExeDir();
        while (!current.empty())
        {
            if (IsStudioRoot(current))
            {
                return SelectStudioRootCandidate(std::move(processOverride), std::move(binding), std::move(savedSetting), current);
            }
            auto parent = current.parent_path();
            if (parent == current) break;
            current = std::move(parent);
        }
        return SelectStudioRootCandidate(std::move(processOverride), std::move(binding), std::move(savedSetting), ExeDir());
    }

    std::filesystem::path DataDir()
    {
        static std::filesystem::path dir = []
        {
            SetLastError(ERROR_SUCCESS);
            DWORD configuredLength = GetEnvironmentVariableW(L"AES_STUDIO_DATA_DIR", nullptr, 0);
            if (configuredLength == 0 && GetLastError() != ERROR_ENVVAR_NOT_FOUND)
            {
                throw winrt::hresult_error(E_INVALIDARG, L"AES_STUDIO_DATA_DIR is set but empty. Set it to an absolute non-root folder path or remove the variable.");
            }
            if (configuredLength > 0)
            {
                std::wstring configuredValue(configuredLength, L'\0');
                DWORD copied = GetEnvironmentVariableW(L"AES_STUDIO_DATA_DIR", configuredValue.data(), configuredLength);
                if (copied == 0 || copied >= configuredLength)
                {
                    throw winrt::hresult_error(E_INVALIDARG, L"AES_STUDIO_DATA_DIR changed while it was being read. Set it to an absolute non-root folder path and restart AES Studio.");
                }
                configuredValue.resize(copied);
                std::filesystem::path configuredPath(configuredValue);
                std::wstring problem;
                if (!IsValidDataDirOverride(configuredPath, problem))
                {
                    throw winrt::hresult_error(E_INVALIDARG, winrt::hstring(problem));
                }
                std::error_code error;
                std::filesystem::create_directories(configuredPath, error);
                if (error || !std::filesystem::is_directory(configuredPath, error) || error)
                {
                    throw winrt::hresult_error(HRESULT_FROM_WIN32(error ? static_cast<DWORD>(error.value()) : ERROR_DIRECTORY),
                        winrt::hstring(L"AES_STUDIO_DATA_DIR could not be created or opened: " + configuredPath.wstring()));
                }
                DWORD writeError = ERROR_SUCCESS;
                if (!ProbeDataDirWritable(configuredPath, writeError))
                {
                    throw winrt::hresult_error(HRESULT_FROM_WIN32(writeError),
                        winrt::hstring(L"AES_STUDIO_DATA_DIR is not writable for AES Studio settings and tokens: " +
                                       configuredPath.wstring() + L". Choose a writable folder or remove the variable."));
                }
                return configuredPath;
            }

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
            auto path = base / L"Programs" / L"AES Studio Data";
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
        constexpr std::wstring_view rootToken = L"%AES_STUDIO_ROOT%";
        size_t position = input.find(rootToken);
        if (position != std::wstring::npos)
        {
            auto root = StudioRoot().wstring();
            do
            {
                input.replace(position, rootToken.size(), root);
                position = input.find(rootToken, position + root.size());
            } while (position != std::wstring::npos);
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
