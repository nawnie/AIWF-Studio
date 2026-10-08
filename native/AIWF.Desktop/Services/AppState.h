// App-wide state every page shares: the project in focus, engine addresses, and saved settings.
//
// One project is in focus at a time (UX principle 2): the title-bar picker and the Projects page
// both set it, and pages that record work (Create, Datasets, Train, Chat context) read it.
// Settings persist in %LOCALAPPDATA%\AIWF Studio\settings.json. The chat key is read from the
// chat engine's own key file each time it is needed and is never stored, shown or logged here.
#pragma once

namespace aiwf
{
    class AppState
    {
    public:
        static AppState& Get();

        // ---- the project in focus (empty id = none chosen yet) ----
        std::wstring ProjectId() const;
        std::wstring ProjectName() const;
        void SetProject(std::wstring id, std::wstring name);   // UI thread; saved; raises ProjectChanged
        winrt::event_token ProjectChanged(std::function<void()> handler);
        void ProjectChanged(winrt::event_token token);

        // ---- engine addresses (from engines.json "endpoints", with built-in defaults) ----
        winrt::hstring EngineApi() const;      // .../api/pro/unified  (the shared engine API)
        winrt::hstring EngineApiRoot() const;  // http://127.0.0.1:7870
        winrt::hstring ChatApi() const;        // http://127.0.0.1:8080  (OpenAI-compatible chat engine)
        winrt::hstring ChatKey() const;        // from the chat engine's key file; empty if missing

        // ---- hand-over between pages: a package picked on Datasets for the Train page ----
        struct PackageChoice
        {
            std::wstring name;
            std::wstring manifestSha256;
        };
        std::optional<PackageChoice> TakePendingPackage();
        void SetPendingPackage(PackageChoice choice);

        // ---- a page asks the main window to switch tasks (keeps the left navigation in step) ----
        void RequestNavigation(std::wstring tag);
        winrt::event_token NavigationRequested(std::function<void(std::wstring const&)> handler);
        void NavigationRequested(winrt::event_token token);

        // ---- guided setup ----
        // false until the setup wizard has been finished (or skipped) once; the window opens on
        // Setup instead of Home while it is false
        bool SetupCompleted() const;
        void MarkSetupCompleted();
        // the AIWF Studio code folder chosen in setup when the app is installed outside it (empty = find it)
        std::wstring StudioRootSetting() const;
        void SetStudioRootSetting(std::wstring folder);
        // folders an engine reads from its environment (Dataset Studio's catalog folder, ReTrain's model
        // folders, ...), chosen in setup. They are added on top of the engine list's own "env" each time
        // the engine starts, so the person's engines.json is never rewritten. Empty value = remove.
        std::vector<std::pair<std::wstring, std::wstring>> EngineEnv(std::wstring const& engineId) const;
        void SetEngineEnv(std::wstring const& engineId, std::wstring const& name, std::wstring value);

        // ---- shutdown ----
        // Set once when the main window closes. Timers, status callbacks and async work that resume
        // after that must not touch XAML any more: the window's content is being torn down, and
        // touching it then crashes inside Microsoft.UI.Xaml.dll. BeginShutdown also drops every
        // ProjectChanged / NavigationRequested handler.
        bool ShuttingDown() const noexcept { return m_shuttingDown.load(); }
        void BeginShutdown();

        // ---- persistence ----
        void Load();
        void Save() const;

    private:
        AppState() = default;
        mutable std::mutex m_lock;
        std::wstring m_projectId;
        std::wstring m_projectName;
        std::optional<PackageChoice> m_pendingPackage;
        bool m_setupCompleted = false;
        std::wstring m_studioRoot;
        std::map<std::wstring, std::map<std::wstring, std::wstring>> m_engineEnv;   // engine id -> name -> value
        std::map<int64_t, std::function<void()>> m_handlers;
        std::map<int64_t, std::function<void(std::wstring const&)>> m_navigationHandlers;
        int64_t m_nextToken = 1;
        std::atomic_bool m_shuttingDown{ false };
    };
}
