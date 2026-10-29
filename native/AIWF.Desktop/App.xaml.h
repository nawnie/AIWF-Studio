// Application object: creates the main window on launch.
#pragma once
#include "App.xaml.g.h"

namespace winrt::AiwfDesktop::implementation
{
    struct App : AppT<App>
    {
        App();
        void OnLaunched(Microsoft::UI::Xaml::LaunchActivatedEventArgs const&);

    private:
        Microsoft::UI::Xaml::Window m_window{ nullptr };
    };
}
