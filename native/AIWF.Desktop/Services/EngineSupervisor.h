// Engine supervisor: starts, watches and stops the local engines the app is built on.
//
// An "engine" is one service the user meets as a single status light (Chat, Images, Datasets,
// Training, and the AIWF engine API that ties them together). Each engine is one or more
// processes described in engines.json, so paths are configuration, not code:
//   1. %LOCALAPPDATA%\AIWF Studio\engines.json  (user override, if present)
//   2. engines.json next to AIWFStudio.exe      (shipped default for this workstation)
//
// Ownership rules (nothing surprising):
//   - Engines this app starts run inside a Windows Job Object with KILL_ON_JOB_CLOSE, so Stop,
//     or closing the app, ends the engine and every child process it spawned. Nothing is left
//     running in the background.
//   - Engines that were already running when the app looked (started by another tool) are
//     shown as "running (started outside AIWF Studio)" and are never stopped by this app.
//   - Processes start with no console window; stdout/stderr go to a log file per engine in
//     %LOCALAPPDATA%\AIWF Studio\logs, which Settings can open.
#pragma once

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
        std::vector<EngineProcessSpec> processes;
    };

    // what the UI shows for one engine at one moment
    struct EngineStatus
    {
        EngineState state = EngineState::Unknown;
        std::wstring detail;         // human sentence: why stopped/failed, or what it is doing
        std::wstring logPath;        // the current or last log file, if this app started it
        std::vector<DWORD> pids;     // processes counted for this engine (own job, or port owners + children)
    };

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
        uint64_t m_nextStartToken = 0;
        uint64_t m_stopGeneration = 0;
        std::filesystem::path m_manifestPath;
        std::map<std::wstring, std::wstring> m_endpoints;
        std::map<int64_t, std::function<void()>> m_handlers;
        int64_t m_nextToken = 1;
    };

    // shared helpers also used by the GPU view (which processes belong to which engine)
    std::vector<DWORD> ListeningPids(std::vector<uint16_t> const& ports);
    std::vector<DWORD> WithDescendants(std::vector<DWORD> const& roots);
    std::wstring StateLabel(EngineState state);
}
