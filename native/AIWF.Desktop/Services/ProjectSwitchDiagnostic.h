#pragma once

#include "Services/Paths.h"
#include <iomanip>
#include <sstream>

namespace aiwf::diagnostics
{
    inline bool ProjectSwitchLoggingEnabled() noexcept
    {
        wchar_t enabled[2]{};
        if (GetEnvironmentVariableW(L"AIWF_STUDIO_PROJECT_SWITCH_DIAG", enabled, 2) != 1 || enabled[0] != L'1')
            return false;

        wchar_t dataDir[2]{};
        return GetEnvironmentVariableW(L"AES_STUDIO_DATA_DIR", dataDir, 2) > 0;
    }

    inline void WriteProjectSwitchLine(std::string const& text) noexcept
    {
        if (!ProjectSwitchLoggingEnabled()) return;
        try
        {
            auto path = aiwf::paths::DataDir() / L"logs" / L"native-project-switch-diagnostic.log";
            std::error_code error;
            std::filesystem::create_directories(path.parent_path(), error);
            if (error) return;

            std::ofstream file(path, std::ios::binary | std::ios::app);
            if (!file) return;
            file << GetTickCount64() << ";thread=" << GetCurrentThreadId() << ';' << text << "\r\n";
            file.flush();
        }
        catch (...)
        {
        }
    }

    inline void MarkProjectSwitchStep(char const* stage) noexcept
    {
        try
        {
            WriteProjectSwitchLine(std::string("stage=") + stage);
        }
        catch (...)
        {
        }
    }

    inline void LogProjectSwitchFailure(char const* stage, winrt::hresult_error const& error) noexcept
    {
        try
        {
            std::ostringstream line;
            auto message = winrt::to_string(error.message());
            line << "failure;stage=" << stage << ";hr=0x" << std::uppercase << std::hex << std::setw(8) << std::setfill('0')
                 << static_cast<uint32_t>(static_cast<HRESULT>(error.code())) << ";message=" << message;
            WriteProjectSwitchLine(line.str());
        }
        catch (...)
        {
            WriteProjectSwitchLine(std::string("failure;stage=") + stage + ";message=<diagnostic formatting failed>");
        }
    }
}
