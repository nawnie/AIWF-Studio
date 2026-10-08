// Settings page: engines (manifest, endpoints, logs), app memory, version, privacy.
#pragma once
#include "SettingsPage.g.h"

namespace winrt::AiwfDesktop::implementation
{
    struct SettingsPage : SettingsPageT<SettingsPage>
    {
        SettingsPage();

        void OnNavigatedTo(Microsoft::UI::Xaml::Navigation::NavigationEventArgs const& args);
        void OnNavigatedFrom(Microsoft::UI::Xaml::Navigation::NavigationEventArgs const& args);

        void OpenManifest_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void RunSetup_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void OpenLogs_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);

    private:
        void ShowEngines();
        void ShowMemory();

        Microsoft::UI::Dispatching::DispatcherQueueTimer m_timer{ nullptr };
        winrt::event_token m_statusToken{};
    };
}

namespace winrt::AiwfDesktop::factory_implementation
{
    struct SettingsPage : SettingsPageT<SettingsPage, implementation::SettingsPage>
    {
    };
}
