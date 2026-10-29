#pragma once
#include <cstdint>
#include <string>
#include <utility>

namespace aiwf
{
    class ProjectChatSession
    {
    public:
        bool BindProject(std::wstring projectId)
        {
            if (m_projectId == projectId) return false;
            m_projectId = std::move(projectId);
            ++m_generation;
            return true;
        }

        std::uint64_t BeginRequest() noexcept { return ++m_generation; }
        void Invalidate() noexcept { ++m_generation; }

        bool IsCurrent(std::uint64_t generation, std::wstring const& projectId) const noexcept
        {
            return generation == m_generation && projectId == m_projectId;
        }

        std::wstring const& ProjectId() const noexcept { return m_projectId; }

    private:
        std::wstring m_projectId;
        std::uint64_t m_generation = 0;
    };
}
