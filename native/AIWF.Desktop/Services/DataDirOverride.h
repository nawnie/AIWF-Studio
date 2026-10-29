#pragma once
#include <algorithm>
#include <filesystem>
#include <string>

namespace aiwf::paths
{
    inline bool IsValidDataDirOverride(std::filesystem::path const& candidate, std::wstring& problem)
    {
        if (candidate.empty())
        {
            problem = L"AES_STUDIO_DATA_DIR is empty.";
            return false;
        }
        if (!candidate.is_absolute())
        {
            problem = L"AES_STUDIO_DATA_DIR must be an absolute path.";
            return false;
        }
        auto normalized = candidate.lexically_normal();
        if (normalized == normalized.root_path())
        {
            problem = L"AES_STUDIO_DATA_DIR must name a folder below the drive or share root.";
            return false;
        }
        auto value = normalized.native();
        if (value.find_first_of(L"*?\"<>|") != std::wstring::npos)
        {
            problem = L"AES_STUDIO_DATA_DIR contains a character that is not allowed in a Windows folder path.";
            return false;
        }
        if (auto colon = value.find(L':'); colon != std::wstring::npos && colon != 1)
        {
            problem = L"AES_STUDIO_DATA_DIR contains a colon outside the drive name.";
            return false;
        }
        for (auto const& component : normalized.relative_path())
        {
            auto part = component.native();
            if (!part.empty() && (part.back() == L'.' || part.back() == L' '))
            {
                problem = L"AES_STUDIO_DATA_DIR has a folder name ending in a dot or space.";
                return false;
            }
            if (std::any_of(part.begin(), part.end(), [](wchar_t character) { return character < 32; }))
            {
                problem = L"AES_STUDIO_DATA_DIR contains a control character.";
                return false;
            }
        }
        problem.clear();
        return true;
    }
}
