// Create page: queue a Qwen Image job, follow it, show the result and recent images.
#pragma once
#include "CreatePage.g.h"

namespace winrt::AiwfDesktop::implementation
{
    struct CreatePage : CreatePageT<CreatePage>
    {
        CreatePage();

        void OnNavigatedTo(Microsoft::UI::Xaml::Navigation::NavigationEventArgs const& args);
        void OnNavigatedFrom(Microsoft::UI::Xaml::Navigation::NavigationEventArgs const& args);

        void Generate_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        fire_and_forget Cancel_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void Dismiss_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void StartEngine_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void Open_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void ShowInFolder_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void Reuse_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);

    private:
        // what the preview is showing, so Open / Show in folder / Use these settings can act on it
        struct Shown
        {
            std::wstring path;
            std::wstring prompt;
            std::wstring aspect;
            int64_t seed = -1;
        };

        fire_and_forget GenerateAsync();
        fire_and_forget LoadRecentAsync();
        fire_and_forget ShowImageAsync(Shown shown, hstring info);
        void UpdateEngineState();
        void SetBusy(bool busy);
        void ApplyLayout(double width);

        // below this page width the form and the result stack into one scrolling column
        static constexpr double StackBelowWidth = 800;
        bool m_stacked = false;
        bool m_layoutKnown = false;

        Shown m_shown;
        std::wstring m_outputDir;     // Studio's output folder, from the engine API
        std::wstring m_jobId;
        bool m_busy = false;
        bool m_disconnected = false;
        bool m_dismissRequested = false;
        bool m_recentLoaded = false;
        winrt::event_token m_statusToken{};
    };
}

namespace winrt::AiwfDesktop::factory_implementation
{
    struct CreatePage : CreatePageT<CreatePage, implementation::CreatePage>
    {
    };
}
