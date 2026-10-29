// Home page: GPU card, engine list with Start/Stop, and the project in focus.
#pragma once
#include "HomePage.g.h"
#include "Services/GpuMonitor.h"

namespace winrt::AiwfDesktop::implementation
{
    struct HomePage : HomePageT<HomePage>
    {
        HomePage();

        void OnNavigatedTo(Microsoft::UI::Xaml::Navigation::NavigationEventArgs const& args);
        void OnNavigatedFrom(Microsoft::UI::Xaml::Navigation::NavigationEventArgs const& args);

        void StartAll_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void StopAll_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void StartApi_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);

    private:
        // one engine row's live parts, kept so a refresh updates text instead of rebuilding rows
        struct EngineRow
        {
            Microsoft::UI::Xaml::Shapes::Ellipse dot{ nullptr };
            Microsoft::UI::Xaml::Controls::TextBlock detail{ nullptr };
            Microsoft::UI::Xaml::Controls::TextBlock state{ nullptr };
            Microsoft::UI::Xaml::Controls::TextBlock vram{ nullptr };
            Microsoft::UI::Xaml::Controls::Button action{ nullptr };
        };

        void BuildEngineRows();
        void UpdateEngineRows();
        void ShowGpu(aiwf::GpuSample const& sample);
        fire_and_forget RefreshAsync();
        fire_and_forget LoadProjectAsync();
        fire_and_forget EngineActionAsync(std::wstring id);

        std::map<std::wstring, EngineRow> m_rows;

        // responsive arrangement of the telemetry tiles and the memory legend
        void ArrangeTiles(double width);
        void ArrangeLegend(double width);
        int m_tileColumns = 0;
        std::vector<Microsoft::UI::Xaml::FrameworkElement> m_legendItems;
        aiwf::GpuSample m_lastSample;
        Microsoft::UI::Dispatching::DispatcherQueueTimer m_timer{ nullptr };
        winrt::event_token m_statusToken{};
        winrt::event_token m_projectToken{};
        bool m_refreshing = false;
        uint64_t m_projectGeneration = 0;
    };
}

namespace winrt::AiwfDesktop::factory_implementation
{
    struct HomePage : HomePageT<HomePage, implementation::HomePage>
    {
    };
}
