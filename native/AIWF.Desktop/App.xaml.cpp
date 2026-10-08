// Application object implementation.
#include "pch.h"
#include "App.xaml.h"
#include "MainWindow.xaml.h"

using namespace winrt;
using namespace winrt::Microsoft::UI::Xaml;

namespace winrt::AiwfDesktop::implementation
{
    App::App()
    {
#if defined _DEBUG && !defined DISABLE_XAML_GENERATED_BREAK_ON_UNHANDLED_EXCEPTION
        // stop in the debugger on unhandled XAML exceptions during development
        UnhandledException([](IInspectable const&, UnhandledExceptionEventArgs const& e)
        {
            if (IsDebuggerPresent())
            {
                auto message = e.Message();
                __debugbreak();
            }
        });
#endif
    }

    void App::OnLaunched([[maybe_unused]] LaunchActivatedEventArgs const& e)
    {
        m_window = make<MainWindow>();
        // Closing the window ends the app right here, after MainWindow's own Closed handler (registered
        // first, so it runs first) has stopped every engine this app started. Settings are already on
        // disk: they are written the moment they change. What is skipped is WinUI's own teardown, which
        // on this Windows App SDK (1.7, Microsoft.UI.Xaml 3.1.7) faults when the last reference to a
        // page or brush happens to drop while the XAML core is shutting down: an access violation in
        // DirectUI::Page::~Page, a WinRT error from the Mica backdrop controller, and a debug-heap
        // use-after-free report in DXamlCore::ShutdownAllPeers, all captured with cdb on 2026-10-07 and
        // all timing-dependent. Ending the process here makes closing deterministic.
        m_window.Closed([this](auto&&, auto&&)
        {
            m_window = nullptr;
            TerminateProcess(GetCurrentProcess(), 0);
        });
        m_window.Activate();
    }
}
