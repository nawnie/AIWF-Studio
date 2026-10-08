// Setup page: the guided first run (this PC, save locations, model locations, engines, review).
//
// Where each answer is saved:
//   - image output, model and checkpoint folders  -> launch.json, through the engine API's
//     POST /setup/folders (the same file Pro's Settings page edits; applies on the next start)
//   - Dataset Studio and ReTrain folders            -> settings.json "engine_env", added to those
//     engines' environment when this app starts them (engines.json itself is never rewritten)
//   - the AIWF Studio code folder (when the app lives elsewhere) -> settings.json "studio_root"
#pragma once
#include "SetupPage.g.h"

namespace winrt::AiwfDesktop::implementation
{
    struct SetupPage : SetupPageT<SetupPage>
    {
        SetupPage();

        void OnNavigatedTo(Microsoft::UI::Xaml::Navigation::NavigationEventArgs const& args);
        void OnNavigatedFrom(Microsoft::UI::Xaml::Navigation::NavigationEventArgs const& args);

        void Back_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void Next_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void Skip_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void Home_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void OpenEngineList_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        fire_and_forget ReloadEngines_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        fire_and_forget Key_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);

    private:
        // one folder question; where its answer is saved depends on engineId (empty = launch.json)
        struct FolderField
        {
            std::wstring key;          // launch.json key, or the engine's environment variable name
            std::wstring engineId;     // "" for launch.json; "datasets" or "training" for engine folders
            std::wstring title;
            std::wstring help;
            bool list = false;         // several folders, one per line
            wchar_t separator = L';';  // how an engine's environment joins a folder list
            int step = 1;              // 1 = save locations, 2 = model locations
            std::wstring original;     // the saved value when the page loaded ("" = default)
            std::wstring fallback;     // the folder(s) used while the box is empty, one per line ("" = unknown)
            std::wstring fallbackFrom; // where that comes from: "the default", "the engine list", ...
            bool available = true;     // false when this app does not start that engine here
            Microsoft::UI::Xaml::Controls::TextBox box{ nullptr };
            Microsoft::UI::Xaml::Controls::TextBlock facts{ nullptr };
        };

        static constexpr int StepCount = 5;

        void ShowStep(int step);
        void ShowPcChecks();
        void BuildFolderRows();
        void UpdateFacts(FolderField& field);
        void LoadEngineFolders();
        fire_and_forget LoadLaunchFoldersAsync();
        void ShowEngines();
        fire_and_forget LoadKeyStateAsync();
        void BuildReview();
        std::wstring CurrentValue(FolderField const& field) const;
        std::vector<FolderField*> ChangedFields();
        fire_and_forget ApplyAsync();
        fire_and_forget ChooseStudioRootAsync();
        fire_and_forget BrowseAsync(size_t index);

        std::vector<FolderField> m_fields;
        int m_step = 0;
        bool m_launchLoaded = false;
        bool m_launchLoading = false;
        bool m_applying = false;
        bool m_applied = false;
        bool m_hasKey = false;
        std::wstring m_launchProblem;
        winrt::event_token m_statusToken{};
    };
}

namespace winrt::AiwfDesktop::factory_implementation
{
    struct SetupPage : SetupPageT<SetupPage, implementation::SetupPage>
    {
    };
}
