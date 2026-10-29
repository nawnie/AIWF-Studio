// Engine supervisor: starts, watches and stops the local engines the app is built on.
//
// An "engine" is one service the user meets as a single status light (Chat, Images, Datasets,
// Training, and the AIWF engine API that ties them together). Each engine is one or more
// processes described in engines.json, so paths are configuration, not code:
//   1. %LOCALAPPDATA%\Programs\AES Studio Data\engines.json  (user override, if present)
//   2. engines.json next to AESStudio.exe      (shipped default for this workstation)
//
// Ownership rules (nothing surprising):
//   - Engines this app starts run inside a Windows Job Object with KILL_ON_JOB_CLOSE, so Stop,
//     or closing the app, ends the engine and every child process it spawned. Nothing is left
//     running in the background.
//   - Engines that were already running when the app looked (started by another tool) are
//     shown as "running (started outside AES Studio)" and are never stopped by this app.
//   - Processes start with no console window; stdout/stderr go to a log file per engine in
//     %LOCALAPPDATA%\Programs\AES Studio Data\logs, which Settings can open.
#pragma once

#include <string_view>

namespace aiwf
{
    enum class EngineState
    {
        Unknown,          // not probed yet
        Stopped,          // nothing answers
        Starting,         // our processes are up, waiting for the ready URL
        Running,          // started by this app and answering
        RunningExternal,  // answering, but started by something else (left alone)
        Failed,           // our start failed or our process exited; Detail says why
        NotConfigured,    // no process command is configured for this engine
    };

    // one process of an engine, in start order
    struct EngineProcessSpec
    {
        std::wstring exe;
        std::vector<std::wstring> args;
        std::wstring cwd;
        std::vector<std::pair<std::wstring, std::wstring>> env;   // added on top of the app's environment
        std::wstring readyUrl;                                     // HTTP answer here = this process is ready
        uint32_t readyTimeoutSeconds = 60;
    };

    struct EngineSpec
    {
        std::wstring id;            // stable key: engine-api, chat, image, datasets, training
        std::wstring name;          // what the user sees
        std::wstring role;          // one line under the name
        std::wstring healthUrl;     // any HTTP answer = running
        std::vector<uint16_t> ports;   // listening ports, used to find external engines' processes
        uint32_t color = 0xFF8A8A8A;   // ARGB used for this engine's slice of the VRAM bar
        bool usesGpu = false;
        bool probeEnabled = true;  // optional external services are not contacted unless enabled
        std::vector<EngineProcessSpec> processes;
    };

    inline bool EngineHasStartCommand(EngineSpec const& engine) noexcept { return !engine.processes.empty(); }

    inline std::wstring EngineApiCompatibilityProblem(bool hasBuildInfo, uint32_t expectedProtocol,
        std::wstring const& expectedFingerprint, bool hasHealthMetadata, uint32_t actualProtocol,
        std::wstring const& actualFingerprint)
    {
        if (!hasBuildInfo)
            return L"This AES Studio build has no valid native-build-info.json. Rebuild or reinstall it with native\\build.ps1 before starting the engine API.";
        if (expectedProtocol != 2)
            return L"This AES Studio build uses an unsupported engine API protocol. Rebuild or reinstall the app and engine API together.";
        if (!hasHealthMetadata)
            return L"The engine API does not report the required compatibility metadata. Update it from the matching AES Studio checkout.";
        if (actualProtocol != expectedProtocol)
            return L"The engine API protocol does not match this AES Studio build. Use the matching app and engine API checkout.";
        if (actualFingerprint != expectedFingerprint)
            return L"The engine API source does not match this AES Studio build. Use the matching app and engine API checkout.";
        return {};
    }

    inline std::wstring EngineApiHealthUrl(std::wstring url)
    {
        constexpr std::wstring_view legacyPath = L"/api/pro/unified/describe";
        auto position = url.rfind(legacyPath);
        if (position != std::wstring::npos && position + legacyPath.size() == url.size())
            url.replace(position, legacyPath.size(), L"/health");
        return url;
    }

    // what the UI shows for one engine at one moment
    struct EngineStatus
    {
        EngineState state = EngineState::Unknown;
        std::wstring detail;         // human sentence: why stopped/failed, or what it is doing
        std::wstring logPath;        // the current or last log file, if this app started it
        std::vector<DWORD> pids;     // processes counted for this engine (own job, or port owners + children)
    };

    inline bool ApplyEngineProbeStatus(EngineStatus& status, bool answers, bool ownAlive, bool requestedStop, bool hasStartCommand)
    {
        if (requestedStop)
        {
            status.state = EngineState::Stopped;
            status.detail = L"Stopped by you.";
            status.pids.clear();
            return false;
        }
        if (answers)
        {
            status.state = ownAlive ? EngineState::Running : EngineState::RunningExternal;
            status.detail = ownAlive ? L"Started by AES Studio." : L"Running; started outside AES Studio, so it is left as is.";
            return false;
        }
        if (ownAlive)
        {
            status.state = EngineState::Starting;
            status.detail = L"Running but not answering yet (loading or busy).";
            return false;
        }
        if (!hasStartCommand)
        {
            status.state = EngineState::NotConfigured;
            status.detail = L"No start command is configured in engines.json.";
            return false;
        }
        if (status.state == EngineState::Running || status.state == EngineState::Starting)
        {
            status.state = EngineState::Failed;
            status.detail = L"Stopped unexpectedly. Open its log in Settings to see why.";
            return true;
        }
        if (status.state != EngineState::Failed)
        {
            status.state = EngineState::Stopped;
            status.detail = L"Not running.";
        }
        return false;
    }

    class EngineSupervisor
    {
    public:
        static EngineSupervisor& Instance();

        // reads engines.json; returns false and fills error when the file is missing or invalid
        bool Load(std::wstring& error);
        std::filesystem::path ManifestPath() const;
        std::vector<EngineSpec> Engines() const;
        EngineSpec const* Find(std::wstring const& id) const;

        // named addresses from the manifest's "endpoints" block (engine_api, chat, chat_key_file),
        // %VARIABLES% expanded; fallback when the manifest does not say
        std::wstring Endpoint(std::wstring const& key, std::wstring const& fallback) const;

        EngineStatus Status(std::wstring const& id) const;
        bool IsAnswering(std::wstring const& id) const;   // Running or RunningExternal

        // probes every engine (background thread); cheap enough to call every couple of seconds
        winrt::Windows::Foundation::IAsyncAction RefreshAsync();

        // starts one engine's processes in order, waiting for each ready URL; no-op if it already answers
        winrt::Windows::Foundation::IAsyncAction StartAsync(std::wstring id, std::optional<uint64_t> expectedGeneration = std::nullopt);

        // stops an engine this app started (terminates its job); external engines are left alone
        bool Stop(std::wstring const& id, std::wstring& message);

        // ends every engine this app started (called when the main window closes)
        void StopAll();
        uint64_t StopGeneration() const;
        bool HasPendingStarts() const;

        // fires on the thread that changed the state; UI code marshals to its dispatcher
        winrt::event_token StatusChanged(std::function<void()> handler);
        void StatusChanged(winrt::event_token token);
        // drops every status handler (the window is closing; pages must not be called back)
        void ClearHandlers();

    private:
        EngineSupervisor() = default;
        struct Owned;   // job handle, process handles, log path for an engine we started

        void SetStatus(std::wstring const& id, EngineState state, std::wstring detail, uint64_t generation, uint64_t startToken);
        std::wstring HealthBearer(std::wstring const& id) const;
        std::wstring DatasetTokenFile() const;
        void RaiseChanged();
        std::vector<DWORD> OwnedPids(std::wstring const& id) const;

        mutable std::mutex m_lock;
        std::vector<EngineSpec> m_engines;
        std::map<std::wstring, EngineStatus> m_status;
        std::map<std::wstring, std::shared_ptr<Owned>> m_owned;
        std::map<std::wstring, uint64_t> m_starting;
        std::map<std::wstring, uint64_t> m_engineGeneration;
        std::set<std::wstring> m_requestedStops;
        uint64_t m_nextStartToken = 0;
        uint64_t m_stopGeneration = 0;
        std::filesystem::path m_manifestPath;
        std::map<std::wstring, std::wstring> m_endpoints;
        uint32_t m_nativeBuildProtocol = 0;
        std::wstring m_apiSourceFingerprint;
        std::wstring m_nativeBuildInfoError;
        std::map<int64_t, std::function<void()>> m_handlers;
        int64_t m_nextToken = 1;
    };

    // shared helpers also used by the GPU view (which processes belong to which engine)
    std::vector<DWORD> ListeningPids(std::vector<uint16_t> const& ports);
    std::vector<DWORD> WithDescendants(std::vector<DWORD> const& roots);
    std::wstring StateLabel(EngineState state);
}
