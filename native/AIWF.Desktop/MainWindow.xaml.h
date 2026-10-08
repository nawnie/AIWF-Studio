// Main window: title bar with the project picker, task navigation, GPU footer, and the
// app-wide refresh loop that keeps engine status current for every page.
#pragma once
#include "MainWindow.g.h"

namespace winrt::AiwfDesktop::implementation
{
    struct MainWindow : MainWindowT<MainWindow>
    {
        MainWindow();

        void Nav_SelectionChanged(Microsoft::UI::Xaml::Controls::NavigationView const& sender,
                                  Microsoft::UI::Xaml::Controls::NavigationViewSelectionChangedEventArgs const& args);
        void ProjectPicker_SelectionChanged(Windows::Foundation::IInspectable const& sender,
                                            Microsoft::UI::Xaml::Controls::SelectionChangedEventArgs const& args);
        // the GPU footer follows the pane: full telemetry when open, one icon when it is an icon rail
        void Nav_DisplayModeChanged(Microsoft::UI::Xaml::Controls::NavigationView const& sender,
                                    Microsoft::UI::Xaml::Controls::NavigationViewDisplayModeChangedEventArgs const& args);
        void Nav_PaneOpening(Microsoft::UI::Xaml::Controls::NavigationView const& sender, Windows::Foundation::IInspectable const& args);
        void Nav_PaneClosing(Microsoft::UI::Xaml::Controls::NavigationView const& sender,
                             Microsoft::UI::Xaml::Controls::NavigationViewPaneClosingEventArgs const& args);
        // narrow title bar: the subtitle gives way to the project picker
        void AppTitleBar_SizeChanged(Windows::Foundation::IInspectable const& sender,
                                     Microsoft::UI::Xaml::SizeChangedEventArgs const& args);

    private:
        void NavigateTo(hstring const& tag);
        void SizeWindow();
        void ShowFooter(bool full);
        fire_and_forget RefreshTick();
        fire_and_forget LoadProjectsAsync();
        fire_and_forget ShowStartupProblemAsync(hstring message);
        void OnClosed(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::WindowEventArgs const& args);

        Microsoft::UI::Dispatching::DispatcherQueueTimer m_timer{ nullptr };
        winrt::event_token m_projectToken{};
        winrt::event_token m_navigationToken{};
        bool m_refreshing = false;
        bool m_loadingProjects = false;
        bool m_projectsLoaded = false;
        bool m_suppressPicker = false;
        bool m_engineApiStartTried = false;
    };
}

namespace winrt::AiwfDesktop::factory_implementation
{
    struct MainWindow : MainWindowT<MainWindow, implementation::MainWindow>
    {
    };
}
