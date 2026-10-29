// Create page: choose an image backend and model, follow the job, and show results.
#pragma once
#include "CreatePage.g.h"

namespace winrt::AiwfDesktop::implementation
{
    struct CreatePage : CreatePageT<CreatePage>
    {
        CreatePage();

        void OnNavigatedTo(Microsoft::UI::Xaml::Navigation::NavigationEventArgs const& args);
        void OnNavigatedFrom(Microsoft::UI::Xaml::Navigation::NavigationEventArgs const& args);

        void ImageBackendPicker_SelectionChanged(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::Controls::SelectionChangedEventArgs const& args);
        void ImageModelPicker_SelectionChanged(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::Controls::SelectionChangedEventArgs const& args);
        void RefreshBackends_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void Generate_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        fire_and_forget Cancel_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void Dismiss_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void StartEngine_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void Open_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void ShowInFolder_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void Reuse_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void CreativePreset_SelectionChanged(Windows::Foundation::IInspectable const&, Microsoft::UI::Xaml::Controls::SelectionChangedEventArgs const&);
        void CreativeRefresh_Click(Windows::Foundation::IInspectable const&, Microsoft::UI::Xaml::RoutedEventArgs const&);
        fire_and_forget CreativeChooseImage_Click(Windows::Foundation::IInspectable const&, Microsoft::UI::Xaml::RoutedEventArgs const&);
        void CreativeRun_Click(Windows::Foundation::IInspectable const&, Microsoft::UI::Xaml::RoutedEventArgs const&);
        fire_and_forget CreativeCancel_Click(Windows::Foundation::IInspectable const&, Microsoft::UI::Xaml::RoutedEventArgs const&);
        void CreativeOpen_Click(Windows::Foundation::IInspectable const&, Microsoft::UI::Xaml::RoutedEventArgs const&);

    private:
        // what the preview is showing, so Open / Show in folder / Use these settings can act on it
        struct Shown
        {
            std::wstring path;
            std::wstring prompt;
            std::wstring aspect;
            std::wstring backendId;
            std::wstring modelId;
            int64_t seed = -1;
        };

        struct ImageModelOption
        {
            std::wstring id;
            std::wstring label;
            std::wstring family;
            std::vector<std::wstring> capabilities;
        };

        struct ImageBackendOption
        {
            std::wstring id;
            std::wstring label;
            std::wstring reason;
            bool available = false;
            std::vector<ImageModelOption> models;
        };

        fire_and_forget GenerateAsync();
        fire_and_forget RefreshCreativeAsync();
        fire_and_forget RunCreativeAsync();
        void CreativeSetBusy(bool busy);
        fire_and_forget LoadBackendsAsync(uint64_t generation);
        fire_and_forget LoadRecentAsync();
        fire_and_forget ShowImageAsync(Shown shown, hstring info);
        ImageBackendOption const* FindBackend(std::wstring const& id) const;
        ImageModelOption const* FindModel(ImageBackendOption const& backend, std::wstring const& id) const;
        void UpdateEngineState();
        void InvalidateBackendDiscovery();
        void StartSelectedBackend();
        void PopulateBackendPicker();
        void PopulateModelPicker();
        void UpdateBackendStatus();
        void UpdateCreateControls();
        void SetBusy(bool busy);
        void ApplyLayout(double width);

        // below this page width the form and the result stack into one scrolling column
        static constexpr double StackBelowWidth = 800;
        bool m_stacked = false;
        bool m_layoutKnown = false;

        bool m_creativeReady = false;
        bool m_creativeBusy = false;
        std::wstring m_creativeImage;
        std::wstring m_creativeResult;
        std::wstring m_creativeJobId;
        std::wstring m_creativeInventoryPreset;
        int m_creativeInventoryPort = 0;
        int m_creativeJobPort = 8188;
        Shown m_shown;
        std::wstring m_outputDir;     // Studio's output folder, from the engine API
        std::wstring m_jobId;
        bool m_busy = false;
        bool m_disconnected = false;
        bool m_dismissRequested = false;
        bool m_recentLoaded = false;
        bool m_backendsLoading = false;
        bool m_backendsLoaded = false;
        bool m_backendLoadAttempted = false;
        bool m_populatingSelectors = false;
        uint64_t m_backendGeneration = 0;
        std::wstring m_backendLoadError;
        std::wstring m_preferenceSaveError;
        std::vector<ImageBackendOption> m_backends;
        winrt::event_token m_statusToken{};
    };
}

namespace winrt::AiwfDesktop::factory_implementation
{
    struct CreatePage : CreatePageT<CreatePage, implementation::CreatePage>
    {
    };
}
