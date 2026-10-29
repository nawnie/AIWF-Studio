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
        constexpr wchar_t DefaultEngineApi[] = L"http://127.0.0.1:18870";
        constexpr wchar_t DefaultChatApi[] = L"http://127.0.0.1:18871";
        constexpr wchar_t DefaultChatKeyFile[] = L"%LOCALAPPDATA%\\Programs\\AES Studio Data\\chat-api-key.txt";

        std::filesystem::path SettingsPath()
        {
            return paths::DataDir() / L"settings.json";
        }

        std::vector<std::wstring> NormalizeImageBackends(std::vector<std::wstring> const& backendIds)
        {
            std::vector<std::wstring> normalized;
            if (std::find(backendIds.begin(), backendIds.end(), L"python") != backendIds.end()) normalized.push_back(L"python");
            if (std::find(backendIds.begin(), backendIds.end(), L"cpp") != backendIds.end()) normalized.push_back(L"cpp");
            return normalized;
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

    bool AppState::SetProject(std::wstring id, std::wstring name)
    {
        std::vector<std::function<void()>> handlers;
        {
            std::lock_guard guard(m_lock);
            if (m_projectId == id && m_projectName == name) return true;
            auto previousId = m_projectId;
            auto previousName = m_projectName;
            m_projectId = std::move(id);
            m_projectName = std::move(name);
            if (!SaveLocked())
            {
                m_projectId = std::move(previousId);
                m_projectName = std::move(previousName);
                return false;
            }
            for (auto const& [token, handler] : m_handlers) handlers.push_back(handler);
        }
        for (auto const& handler : handlers) handler();
        return true;
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

    // ---- native Create preferences ----------------------------------------------------------------------
    std::vector<std::wstring> AppState::EnabledImageBackends() const
    {
        std::lock_guard guard(m_lock);
        return m_enabledImageBackends;
    }

    bool AppState::IsImageBackendEnabled(std::wstring const& backendId) const
    {
        std::lock_guard guard(m_lock);
        return std::find(m_enabledImageBackends.begin(), m_enabledImageBackends.end(), backendId) != m_enabledImageBackends.end();
    }

    bool AppState::SetEnabledImageBackends(std::vector<std::wstring> backendIds)
    {
        auto normalized = NormalizeImageBackends(backendIds);
        if (normalized.empty()) return false;
        std::lock_guard guard(m_lock);
        auto previousBackends = m_enabledImageBackends;
        auto previousBackendId = m_imageBackendId;
        if (m_enabledImageBackends == normalized &&
            std::find(normalized.begin(), normalized.end(), m_imageBackendId) != normalized.end()) return SaveLocked();
        m_enabledImageBackends = normalized;
        if (std::find(normalized.begin(), normalized.end(), m_imageBackendId) == normalized.end()) m_imageBackendId = normalized.front();
        if (SaveLocked()) return true;
        m_enabledImageBackends = std::move(previousBackends);
        m_imageBackendId = std::move(previousBackendId);
        return false;
    }

    std::wstring AppState::ImageBackendId() const
    {
        std::lock_guard guard(m_lock);
        return m_imageBackendId;
    }

    std::wstring AppState::ImageModelId(std::wstring const& backendId) const
    {
        std::lock_guard guard(m_lock);
        auto found = m_imageModelIds.find(backendId);
        return found == m_imageModelIds.end() ? std::wstring{} : found->second;
    }

    bool AppState::SetImageBackendId(std::wstring id)
    {
        if (id.empty()) return false;
        std::lock_guard guard(m_lock);
        if (std::find(m_enabledImageBackends.begin(), m_enabledImageBackends.end(), id) == m_enabledImageBackends.end()) return false;
        if (m_imageBackendId == id) return SaveLocked();
        auto previous = m_imageBackendId;
        m_imageBackendId = std::move(id);
        if (SaveLocked()) return true;
        m_imageBackendId = std::move(previous);
        return false;
    }

    bool AppState::SetImageModelId(std::wstring const& backendId, std::wstring id)
    {
        if (backendId.empty()) return false;
        std::lock_guard guard(m_lock);
        if (std::find(m_enabledImageBackends.begin(), m_enabledImageBackends.end(), backendId) == m_enabledImageBackends.end()) return false;
        auto found = m_imageModelIds.find(backendId);
        if ((found == m_imageModelIds.end() && id.empty()) || (found != m_imageModelIds.end() && found->second == id)) return SaveLocked();
        std::optional<std::wstring> previous;
        if (found != m_imageModelIds.end()) previous = found->second;
        if (id.empty()) m_imageModelIds.erase(backendId);
        else m_imageModelIds[backendId] = std::move(id);
        if (SaveLocked()) return true;
        if (previous) m_imageModelIds[backendId] = std::move(*previous);
        else m_imageModelIds.erase(backendId);
        return false;
    }

    bool AppState::SetImageSelection(std::wstring backendId, std::wstring modelId)
    {
        if (backendId.empty() || modelId.empty()) return false;
        std::lock_guard guard(m_lock);
        if (std::find(m_enabledImageBackends.begin(), m_enabledImageBackends.end(), backendId) == m_enabledImageBackends.end()) return false;
        auto found = m_imageModelIds.find(backendId);
        if (found != m_imageModelIds.end() && m_imageBackendId == backendId && found->second == modelId) return SaveLocked();
        auto previousBackend = m_imageBackendId;
        std::optional<std::wstring> previousModel;
        if (found != m_imageModelIds.end()) previousModel = found->second;
        m_imageBackendId = backendId;
        m_imageModelIds[backendId] = std::move(modelId);
        if (SaveLocked()) return true;
        m_imageBackendId = std::move(previousBackend);
        if (previousModel) m_imageModelIds[backendId] = std::move(*previousModel);
        else m_imageModelIds.erase(backendId);
        return false;
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

    bool AppState::MarkSetupCompleted()
    {
        std::lock_guard guard(m_lock);
        if (m_setupCompleted) return SaveLocked();
        m_setupCompleted = true;
        if (SaveLocked()) return true;
        m_setupCompleted = false;
        return false;
    }

    std::wstring AppState::StudioRootSetting() const
    {
        std::lock_guard guard(m_lock);
        return m_studioRoot;
    }

    bool AppState::SetStudioRootSetting(std::wstring folder)
    {
        std::wstring problem;
        if (!paths::ValidateStudioRootSelection(std::filesystem::path(folder), problem)) return false;
        std::lock_guard guard(m_lock);
        if (m_studioRoot == folder) return SaveLocked();
        auto previous = m_studioRoot;
        m_studioRoot = folder;
        if (!SaveLocked())
        {
            m_studioRoot = std::move(previous);
            return false;
        }
        paths::SetStudioRootOverride(folder);
        return true;
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

    bool AppState::SetEngineEnv(std::wstring const& engineId, std::wstring const& name, std::wstring value)
    {
        if (engineId.empty() || name.empty()) return false;
        std::lock_guard guard(m_lock);
        std::optional<std::wstring> previous;
        auto found = m_engineEnv.find(engineId);
        if (found != m_engineEnv.end())
        {
            auto variable = found->second.find(name);
            if (variable != found->second.end()) previous = variable->second;
        }
        if (value.empty())
        {
            if (found != m_engineEnv.end())
            {
                found->second.erase(name);
                if (found->second.empty()) m_engineEnv.erase(found);
            }
        }
        else m_engineEnv[engineId][name] = std::move(value);
        if (SaveLocked()) return true;
        if (previous) m_engineEnv[engineId][name] = std::move(*previous);
        else if (auto current = m_engineEnv.find(engineId); current != m_engineEnv.end())
        {
            current->second.erase(name);
            if (current->second.empty()) m_engineEnv.erase(current);
        }
        return false;
    }

    bool AppState::ApplySetupSettings(std::vector<std::wstring> backendIds, std::vector<EngineEnvUpdate> engineEnvUpdates)
    {
        auto normalized = NormalizeImageBackends(backendIds);
        if (normalized.empty()) return false;
        std::lock_guard guard(m_lock);
        auto previousBackends = m_enabledImageBackends;
        auto previousBackendId = m_imageBackendId;
        auto previousEngineEnv = m_engineEnv;
        bool previousSetupCompleted = m_setupCompleted;

        m_enabledImageBackends = std::move(normalized);
        if (std::find(m_enabledImageBackends.begin(), m_enabledImageBackends.end(), m_imageBackendId) == m_enabledImageBackends.end())
            m_imageBackendId = m_enabledImageBackends.front();
        for (auto const& update : engineEnvUpdates)
        {
            if (update.engineId.empty() || update.name.empty()) continue;
            if (update.value.empty())
            {
                if (auto engine = m_engineEnv.find(update.engineId); engine != m_engineEnv.end())
                {
                    engine->second.erase(update.name);
                    if (engine->second.empty()) m_engineEnv.erase(engine);
                }
            }
            else m_engineEnv[update.engineId][update.name] = update.value;
        }
        m_setupCompleted = true;
        if (SaveLocked()) return true;
        m_enabledImageBackends = std::move(previousBackends);
        m_imageBackendId = std::move(previousBackendId);
        m_engineEnv = std::move(previousEngineEnv);
        m_setupCompleted = previousSetupCompleted;
        return false;
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
        auto path = SettingsPath();
        auto text = paths::ReadUtf8(path);
        JsonObject root{ nullptr };
        if (!text || !JsonObject::TryParse(to_hstring(*text), root))
        {
            std::error_code error;
            bool exists = std::filesystem::exists(path, error);
            if (text || exists || error)
            {
                std::lock_guard guard(m_lock);
                m_settingsLoadFailed = true;
            }
            return;
        }
        std::wstring studioRoot;
        {
            std::lock_guard guard(m_lock);
            if (auto value = root.TryLookup(L"project_id"); value && value.ValueType() == JsonValueType::String) m_projectId = value.GetString();
            if (auto value = root.TryLookup(L"project_name"); value && value.ValueType() == JsonValueType::String) m_projectName = value.GetString();
            if (auto value = root.TryLookup(L"setup_completed"); value && value.ValueType() == JsonValueType::Boolean) m_setupCompleted = value.GetBoolean();
            if (auto value = root.TryLookup(L"studio_root"); value && value.ValueType() == JsonValueType::String) m_studioRoot = value.GetString();
            if (auto value = root.TryLookup(L"enabled_image_backends"); value && value.ValueType() == JsonValueType::Array)
            {
                std::vector<std::wstring> enabled;
                for (auto const& backend : value.GetArray())
                {
                    if (backend.ValueType() == JsonValueType::String) enabled.push_back(std::wstring(backend.GetString()));
                }
                auto normalized = NormalizeImageBackends(enabled);
                m_enabledImageBackends = normalized.empty() ? std::vector<std::wstring>{ L"cpp" } : std::move(normalized);
            }
            if (auto value = root.TryLookup(L"image_backend_id"); value && value.ValueType() == JsonValueType::String && !value.GetString().empty())
            {
                m_imageBackendId = value.GetString();
            }
            if (std::find(m_enabledImageBackends.begin(), m_enabledImageBackends.end(), m_imageBackendId) == m_enabledImageBackends.end())
            {
                m_imageBackendId = m_enabledImageBackends.front();
            }
            if (auto value = root.TryLookup(L"image_model_ids"); value && value.ValueType() == JsonValueType::Object)
            {
                for (auto const& model : value.GetObject())
                {
                    if (model.Value().ValueType() == JsonValueType::String && !model.Value().GetString().empty())
                    {
                        m_imageModelIds[std::wstring(model.Key())] = model.Value().GetString();
                    }
                }
            }
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

    bool AppState::Save() const
    {
        std::lock_guard guard(m_lock);
        return SaveLocked();
    }

    bool AppState::SaveLocked() const
    {
        if (m_settingsLoadFailed) return false;
        JsonObject root;
        root.Insert(L"schema", JsonValue::CreateStringValue(L"aiwf-desktop-settings-v1"));
        root.Insert(L"project_id", JsonValue::CreateStringValue(m_projectId));
        root.Insert(L"project_name", JsonValue::CreateStringValue(m_projectName));
        root.Insert(L"setup_completed", JsonValue::CreateBooleanValue(m_setupCompleted));
        if (!m_studioRoot.empty()) root.Insert(L"studio_root", JsonValue::CreateStringValue(m_studioRoot));
        JsonArray enabledBackends;
        for (auto const& backendId : m_enabledImageBackends)
        {
            enabledBackends.Append(JsonValue::CreateStringValue(backendId));
        }
        root.Insert(L"enabled_image_backends", enabledBackends);
        root.Insert(L"image_backend_id", JsonValue::CreateStringValue(m_imageBackendId));
        if (!m_imageModelIds.empty())
        {
            JsonObject models;
            for (auto const& [backendId, modelId] : m_imageModelIds)
            {
                models.Insert(backendId, JsonValue::CreateStringValue(modelId));
            }
            root.Insert(L"image_model_ids", models);
        }
        // this loop writes the per-engine folder settings back in the same shape Load reads
        JsonObject engines;
        for (auto const& [engineId, variables] : m_engineEnv)
        {
            JsonObject values;
            for (auto const& [name, value] : variables) values.Insert(name, JsonValue::CreateStringValue(value));
            engines.Insert(engineId, values);
        }
        if (engines.Size() > 0) root.Insert(L"engine_env", engines);
        return paths::WriteUtf8Atomic(SettingsPath(), to_string(root.Stringify()));
    }
}
