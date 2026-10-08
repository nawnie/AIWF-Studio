// Settings page implementation.
#include "pch.h"
#include <psapi.h>
#include "SettingsPage.xaml.h"
#if __has_include("SettingsPage.g.cpp")
#include "SettingsPage.g.cpp"
#endif
#include "Services/AppState.h"
#include "Services/EngineSupervisor.h"
#include "Services/Paths.h"
#include "Services/Ui.h"

using namespace winrt;
using namespace winrt::Windows::Foundation;
using namespace winrt::Microsoft::UI::Xaml;
using namespace winrt::Microsoft::UI::Xaml::Controls;
using namespace winrt::Microsoft::UI::Xaml::Navigation;
namespace ui = aiwf::ui;

namespace winrt::AiwfDesktop::implementation
{
    SettingsPage::SettingsPage()
    {
        InitializeComponent();
        VersionText().Text(L"AIWF Studio for Windows 0.1 (MVP)  ·  C++/WinRT and WinUI 3 on the Windows App SDK 1.7, built with MSVC  ·  "
                           L"no browser engine in the app itself");
        DataText().Text(hstring(L"Settings and logs: " + aiwf::paths::DataDir().wstring()));
    }

    void SettingsPage::OnNavigatedTo(NavigationEventArgs const&)
    {
        auto weak = get_weak();
        auto dispatcher = DispatcherQueue();
        m_statusToken = aiwf::EngineSupervisor::Instance().StatusChanged([weak, dispatcher]
        {
            dispatcher.TryEnqueue([weak]
            {
                if (auto self = weak.get(); self && !aiwf::AppState::Get().ShuttingDown()) self->ShowEngines();
            });
        });
        m_timer = dispatcher.CreateTimer();
        m_timer.Interval(std::chrono::seconds(2));
        m_timer.Tick([weak](auto&&, auto&&)
        {
            if (auto self = weak.get(); self && !aiwf::AppState::Get().ShuttingDown()) self->ShowMemory();
        });
        m_timer.Start();
        ShowEngines();
        ShowMemory();
    }

    void SettingsPage::OnNavigatedFrom(NavigationEventArgs const&)
    {
        if (m_timer) m_timer.Stop();
        aiwf::EngineSupervisor::Instance().StatusChanged(m_statusToken);
    }

    // ---- engines: where each comes from, its state and its log ------------------------------------------------
    void SettingsPage::ShowEngines()
    {
        auto& supervisor = aiwf::EngineSupervisor::Instance();
        ManifestText().Text(hstring(L"Engine list: " + supervisor.ManifestPath().wstring() +
                                    L"   (a copy in " + (aiwf::paths::DataDir() / L"engines.json").wstring() + L" overrides it for this user)"));
        EngineRows().Children().Clear();
        for (auto const& engine : supervisor.Engines())
        {
            auto status = supervisor.Status(engine.id);
            Grid row;
            row.ColumnSpacing(12);
            ColumnDefinition textColumn;
            textColumn.Width(GridLengthHelper::FromValueAndType(1, GridUnitType::Star));
            ColumnDefinition buttonColumn;
            buttonColumn.Width(GridLengthHelper::Auto());
            row.ColumnDefinitions().Append(textColumn);
            row.ColumnDefinitions().Append(buttonColumn);

            StackPanel text;
            text.Children().Append(ui::Text(hstring(engine.name + L"  ·  " + aiwf::StateLabel(status.state)), L"BodyStrongTextBlockStyle"));
            auto address = ui::Text(hstring(engine.healthUrl), L"CaptionTextBlockStyle", 0.7);
            address.IsTextSelectionEnabled(true);
            text.Children().Append(address);
            if (!status.detail.empty()) text.Children().Append(ui::Text(hstring(status.detail), L"CaptionTextBlockStyle", 0.6));
            row.Children().Append(text);

            if (!status.logPath.empty())
            {
                Button open;
                open.Content(box_value(L"Open log"));
                open.VerticalAlignment(VerticalAlignment::Center);
                open.Click([path = status.logPath](auto&&, auto&&) { ui::OpenWithShell(path); });
                Grid::SetColumn(open, 1);
                row.Children().Append(open);
            }
            EngineRows().Children().Append(row);
        }
    }

    void SettingsPage::OpenManifest_Click(IInspectable const&, RoutedEventArgs const&)
    {
        ui::ShowInExplorer(aiwf::EngineSupervisor::Instance().ManifestPath().wstring());
    }

    // the guided setup again (folders and engines); it has no menu item of its own
    void SettingsPage::RunSetup_Click(IInspectable const&, RoutedEventArgs const&)
    {
        aiwf::AppState::Get().RequestNavigation(L"setup");
    }

    void SettingsPage::OpenLogs_Click(IInspectable const&, RoutedEventArgs const&)
    {
        ui::OpenWithShell(aiwf::paths::LogsDir().wstring());
    }

    // ---- what the app itself costs ---------------------------------------------------------------------------------
    void SettingsPage::ShowMemory()
    {
        PROCESS_MEMORY_COUNTERS_EX counters{};
        counters.cb = sizeof(counters);
        if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters)))
        {
            MemoryText().Text(hstring(L"This window uses " + aiwf::paths::FormatBytes(counters.WorkingSetSize) + L" of RAM (" +
                                      aiwf::paths::FormatBytes(counters.PrivateUsage) + L" private). The engines' memory is shown on Home."));
        }
    }
}
