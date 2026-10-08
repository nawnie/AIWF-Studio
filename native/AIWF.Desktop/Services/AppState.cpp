// App-wide state implementation (see AppState.h).
#include "pch.h"
#include "Services/AppState.h"
#include "Services/EngineSupervisor.h"
#include "Services/Paths.h"

using namespace winrt;
using namespace winrt::Windows::Data::Json;

namespace aiwf
{
    namespace
    {
        constexpr wchar_t DefaultEngineApi[] = L"http://127.0.0.1:7870";
        constexpr wchar_t DefaultChatApi[] = L"http://127.0.0.1:8080";
        constexpr wchar_t DefaultChatKeyFile[] = L"%USERPROFILE%\\.llama-chat\\api-key.txt";

        std::filesystem::path SettingsPath()
        {
            return paths::DataDir() / L"settings.json";
        }
    }

    AppState& AppState::Get()
    {
        static AppState* instance = new AppState();   // lives for the whole process
        return *instance;
    }

    // ---- the project in focus ----------------------------------------------------------------------
    std::wstring AppState::ProjectId() const
    {
        std::lock_guard guard(m_lock);
        return m_projectId;
    }

    std::wstring AppState::ProjectName() const
    {
        std::lock_guard guard(m_lock);
        return m_projectName;
    }

    void AppState::SetProject(std::wstring id, std::wstring name)
    {
        std::vector<std::function<void()>> handlers;
        {
            std::lock_guard guard(m_lock);
            if (m_projectId == id && m_projectName == name) return;
            m_projectId = std::move(id);
            m_projectName = std::move(name);
            for (auto const& [token, handler] : m_handlers) handlers.push_back(handler);
        }
        Save();
        for (auto const& handler : handlers) handler();
    }

    event_token AppState::ProjectChanged(std::function<void()> handler)
    {
        std::lock_guard guard(m_lock);
        event_token token{ m_nextToken++ };
        m_handlers[token.value] = std::move(handler);
        return token;
    }

    void AppState::ProjectChanged(event_token token)
    {
        std::lock_guard guard(m_lock);
        m_handlers.erase(token.value);
    }

    // ---- engine addresses ---------------------------------------------------------------------------
    hstring AppState::EngineApiRoot() const
    {
        return hstring(EngineSupervisor::Instance().Endpoint(L"engine_api", DefaultEngineApi));
    }

    hstring AppState::EngineApi() const
    {
        return EngineApiRoot() + L"/api/pro/unified";
    }

    hstring AppState::ChatApi() const
    {
        return hstring(EngineSupervisor::Instance().Endpoint(L"chat", DefaultChatApi));
    }

    hstring AppState::ChatKey() const
    {
        auto file = EngineSupervisor::Instance().Endpoint(L"chat_key_file", paths::Expand(DefaultChatKeyFile));
        return hstring(paths::FirstKeyLine(file));
    }

    // ---- page hand-over --------------------------------------------------------------------------------
    std::optional<AppState::PackageChoice> AppState::TakePendingPackage()
    {
        std::lock_guard guard(m_lock);
        auto choice = std::move(m_pendingPackage);
        m_pendingPackage.reset();
        return choice;
    }

    void AppState::SetPendingPackage(PackageChoice choice)
    {
        std::lock_guard guard(m_lock);
        m_pendingPackage = std::move(choice);
    }

    // ---- navigation requests ----------------------------------------------------------------------------
    void AppState::RequestNavigation(std::wstring tag)
    {
        std::vector<std::function<void(std::wstring const&)>> handlers;
        {
            std::lock_guard guard(m_lock);
            for (auto const& [token, handler] : m_navigationHandlers) handlers.push_back(handler);
        }
        for (auto const& handler : handlers) handler(tag);
    }

    event_token AppState::NavigationRequested(std::function<void(std::wstring const&)> handler)
    {
        std::lock_guard guard(m_lock);
        event_token token{ m_nextToken++ };
        m_navigationHandlers[token.value] = std::move(handler);
        return token;
    }

    void AppState::NavigationRequested(event_token token)
    {
        std::lock_guard guard(m_lock);
        m_navigationHandlers.erase(token.value);
    }

    // ---- guided setup -------------------------------------------------------------------------------------
    bool AppState::SetupCompleted() const
    {
        std::lock_guard guard(m_lock);
        return m_setupCompleted;
    }

    void AppState::MarkSetupCompleted()
    {
        {
            std::lock_guard guard(m_lock);
            m_setupCompleted = true;
        }
        Save();
    }

    std::wstring AppState::StudioRootSetting() const
    {
        std::lock_guard guard(m_lock);
        return m_studioRoot;
    }

    void AppState::SetStudioRootSetting(std::wstring folder)
    {
        {
            std::lock_guard guard(m_lock);
            m_studioRoot = folder;
        }
        paths::SetStudioRootOverride(folder);
        Save();
    }

    std::vector<std::pair<std::wstring, std::wstring>> AppState::EngineEnv(std::wstring const& engineId) const
    {
        std::lock_guard guard(m_lock);
        std::vector<std::pair<std::wstring, std::wstring>> values;
        if (auto found = m_engineEnv.find(engineId); found != m_engineEnv.end())
        {
            for (auto const& [name, value] : found->second) values.emplace_back(name, value);
        }
        return values;
    }

    void AppState::SetEngineEnv(std::wstring const& engineId, std::wstring const& name, std::wstring value)
    {
        {
            std::lock_guard guard(m_lock);
            if (value.empty())
            {
                if (auto found = m_engineEnv.find(engineId); found != m_engineEnv.end())
                {
                    found->second.erase(name);
                    if (found->second.empty()) m_engineEnv.erase(found);
                }
            }
            else
            {
                m_engineEnv[engineId][name] = std::move(value);
            }
        }
        Save();
    }

    // ---- shutdown ---------------------------------------------------------------------------------------
    void AppState::BeginShutdown()
    {
        m_shuttingDown = true;
        std::lock_guard guard(m_lock);
        m_handlers.clear();
        m_navigationHandlers.clear();
    }

    // ---- persistence ---------------------------------------------------------------------------------------
    void AppState::Load()
    {
        auto text = paths::ReadUtf8(SettingsPath());
        JsonObject root{ nullptr };
        if (!text || !JsonObject::TryParse(to_hstring(*text), root)) return;
        std::wstring studioRoot;
        {
            std::lock_guard guard(m_lock);
            if (auto value = root.TryLookup(L"project_id"); value && value.ValueType() == JsonValueType::String) m_projectId = value.GetString();
            if (auto value = root.TryLookup(L"project_name"); value && value.ValueType() == JsonValueType::String) m_projectName = value.GetString();
            if (auto value = root.TryLookup(L"setup_completed"); value && value.ValueType() == JsonValueType::Boolean) m_setupCompleted = value.GetBoolean();
            if (auto value = root.TryLookup(L"studio_root"); value && value.ValueType() == JsonValueType::String) m_studioRoot = value.GetString();
            // this loop reads the per-engine folder settings: {"datasets": {"DATASET_STUDIO_STATE": "D:\\..."}}
            if (auto value = root.TryLookup(L"engine_env"); value && value.ValueType() == JsonValueType::Object)
            {
                for (auto const& engine : value.GetObject())
                {
                    if (engine.Value().ValueType() != JsonValueType::Object) continue;
                    for (auto const& variable : engine.Value().GetObject())
                    {
                        if (variable.Value().ValueType() == JsonValueType::String && !variable.Value().GetString().empty())
                        {
                            m_engineEnv[std::wstring(engine.Key())][std::wstring(variable.Key())] = variable.Value().GetString();
                        }
                    }
                }
            }
            studioRoot = m_studioRoot;
        }
        // the engine list expands %AIWF_STUDIO_ROOT% from this, so it must be known before engines load
        paths::SetStudioRootOverride(studioRoot);
    }

    void AppState::Save() const
    {
        JsonObject root;
        {
            std::lock_guard guard(m_lock);
            root.Insert(L"schema", JsonValue::CreateStringValue(L"aiwf-desktop-settings-v1"));
            root.Insert(L"project_id", JsonValue::CreateStringValue(m_projectId));
            root.Insert(L"project_name", JsonValue::CreateStringValue(m_projectName));
            root.Insert(L"setup_completed", JsonValue::CreateBooleanValue(m_setupCompleted));
            if (!m_studioRoot.empty()) root.Insert(L"studio_root", JsonValue::CreateStringValue(m_studioRoot));
            // this loop writes the per-engine folder settings back in the same shape Load reads
            JsonObject engines;
            for (auto const& [engineId, variables] : m_engineEnv)
            {
                JsonObject values;
                for (auto const& [name, value] : variables) values.Insert(name, JsonValue::CreateStringValue(value));
                engines.Insert(engineId, values);
            }
            if (engines.Size() > 0) root.Insert(L"engine_env", engines);
        }
        paths::WriteUtf8Atomic(SettingsPath(), to_string(root.Stringify()));
    }
}
