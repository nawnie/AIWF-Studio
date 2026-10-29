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
        try
        {
            m_window = make<MainWindow>();
        }
        catch (hresult_error const& error)
        {
            auto message = std::wstring(error.message());
            MessageBoxW(nullptr, message.c_str(), L"AES Studio could not start", MB_OK | MB_ICONERROR);
            TerminateProcess(GetCurrentProcess(), 1);
            return;
        }
        // Closing the window ends the app right here, after MainWindow's own Closed handler (registered
        // first, so it runs first) has stopped every engine this app started. Settings are already on
        // disk: they are written the moment they change. The original 2026-10-07 build skipped WinUI's
        // teardown after timing-dependent faults on Windows App SDK 1.7 / Microsoft.UI.Xaml 3.1.7 when
        // the last reference to a page or brush dropped while the XAML core was shutting down: an access violation in
        // DirectUI::Page::~Page, a WinRT error from the Mica backdrop controller, and a debug-heap
        // use-after-free report in DXamlCore::ShutdownAllPeers. The reports were captured with cdb on
        // 2026-10-07 against Windows App SDK 1.7 / Microsoft.UI.Xaml 3.1.7 and are timing-dependent.
        // This source now builds with Windows App SDK 2.5.1 / WinUI 3.2.3; the teardown workaround has
        // not been revalidated on that version, so it stays in place until teardown testing is repeated.
        // Ending the process here makes closing deterministic.
        m_window.Closed([this](auto&&, auto&&)
        {
            m_window = nullptr;
            TerminateProcess(GetCurrentProcess(), 0);
        });
        m_window.Activate();
    }
}
