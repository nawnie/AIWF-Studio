// Main window implementation.
#include "pch.h"
#include "MainWindow.xaml.h"
#if __has_include("MainWindow.g.cpp")
#include "MainWindow.g.cpp"
#endif
#include "HomePage.xaml.h"
#include "ChatPage.xaml.h"
#include "CreatePage.xaml.h"
#include "DatasetsPage.xaml.h"
#include "TrainPage.xaml.h"
#include "ProjectsPage.xaml.h"
#include "SettingsPage.xaml.h"
#include "SetupPage.xaml.h"
#include "Services/AppState.h"
#include "Services/EngineSupervisor.h"
#include "Services/GpuMonitor.h"
#include "Services/Http.h"
#include "Services/Paths.h"
#include "Services/ProjectPickerPolicy.h"
#include "Services/ProjectSwitchDiagnostic.h"
#include "Services/Ui.h"

using namespace winrt;
using namespace winrt::Windows::Foundation;
using namespace winrt::Windows::Data::Json;
using namespace winrt::Microsoft::UI::Xaml;
using namespace winrt::Microsoft::UI::Xaml::Controls;
namespace ui = aiwf::ui;

namespace winrt::AiwfDesktop::implementation
{
    MainWindow::MainWindow()
    {
        InitializeComponent();

        // The settings entry is called "Configure Studio". NavigationView creates its settings item only
        // when its template is applied, so the label is set again once the view has loaded.
        auto renameSettings = [](NavigationView const& nav)
        {
            if (auto settingsItem = nav.SettingsItem().try_as<NavigationViewItem>())
            {
                settingsItem.Content(box_value(L"Configure Studio"));
                Microsoft::UI::Xaml::Automation::AutomationProperties::SetName(settingsItem, L"Configure Studio");
            }
        };
        renameSettings(Nav());
        Nav().Loaded([renameSettings](IInspectable const& sender, RoutedEventArgs const&)
        {
            renameSettings(sender.as<NavigationView>());
        });

        // the content area extends under the title bar; AppTitleBar becomes the drag region
        ExtendsContentIntoTitleBar(true);
        SetTitleBar(AppTitleBar());
        SizeWindow();
        // title bar and taskbar icon (the exe's own icon resource covers Explorer and shortcuts)
        auto icon = aiwf::paths::ExeDir() / L"Assets" / L"AIWF.ico";
        if (std::filesystem::exists(icon)) AppWindow().SetIcon(icon.wstring());

        // this block loads saved settings and the engine list before any page asks for them
        aiwf::AppState::Get().Load();
        std::wstring problem;
        if (!aiwf::EngineSupervisor::Instance().Load(problem))
        {
            ShowStartupProblemAsync(hstring(problem));
        }

        // the picker follows project changes made anywhere (Projects page, a new project, ...)
        auto weak = get_weak();
        m_projectToken = aiwf::AppState::Get().ProjectChanged([weak]
        {
            if (auto self = weak.get(); self && !aiwf::AppState::Get().ShuttingDown())
            {
                aiwf::diagnostics::MarkProjectSwitchStep("main.project-changed-handler");
                self->SyncProjectPickerToState();
            }
        });
        // a page asking for another task selects that item, which also navigates
        m_navigationToken = aiwf::AppState::Get().NavigationRequested([weak](std::wstring const& tag)
        {
            auto self = weak.get();
            if (!self || aiwf::AppState::Get().ShuttingDown()) return;
            // Setup has no menu item: clear the selection and show it directly
            if (tag == L"setup")
            {
                self->Nav().SelectedItem(nullptr);
                self->NavigateTo(L"setup");
                return;
            }
            for (auto const& entry : self->Nav().MenuItems())
            {
                auto item = entry.try_as<NavigationViewItem>();
                if (item && unbox_value_or<hstring>(item.Tag(), L"") == tag) self->Nav().SelectedItem(item);
            }
        });
        Closed({ this, &MainWindow::OnClosed });

        // the app-wide refresh: engine status for every page, and the GPU line in the footer
        m_timer = DispatcherQueue().CreateTimer();
        m_timer.Interval(std::chrono::seconds(2));
        m_timer.Tick([weak](auto&&, auto&&)
        {
            if (auto self = weak.get(); self && !aiwf::AppState::Get().ShuttingDown()) self->RefreshTick();
        });
        m_timer.Start();
        RefreshTick();

        // the first run opens the guided setup; after that the app starts on Home
        if (aiwf::AppState::Get().SetupCompleted()) Nav().SelectedItem(Nav().MenuItems().GetAt(0));
        else NavigateTo(L"setup");
    }

    // Opens at 1280 x 820 layout units, centered. The work area is in physical pixels, so the size is
    // scaled by the monitor's DPI; on a small or highly scaled screen (where that would not fit with a
    // margin) the window opens maximized instead of as a cramped window.
    void MainWindow::SizeWindow()
    {
        auto appWindow = AppWindow();
        auto area = Microsoft::UI::Windowing::DisplayArea::GetFromWindowId(appWindow.Id(), Microsoft::UI::Windowing::DisplayAreaFallback::Nearest);
        auto work = area.WorkArea();
        auto hwnd = Microsoft::UI::GetWindowFromWindowId(appWindow.Id());
        double scale = (hwnd ? GetDpiForWindow(hwnd) : 96) / 96.0;
        // the smallest size every page is laid out for (560 x 460 layout units); below that, controls
        // would overlap, so Windows is asked not to let the window shrink further
        if (auto presenter = appWindow.Presenter().try_as<Microsoft::UI::Windowing::OverlappedPresenter>())
        {
            presenter.PreferredMinimumWidth(static_cast<int32_t>(560 * scale));
            presenter.PreferredMinimumHeight(static_cast<int32_t>(460 * scale));
        }
        auto width = static_cast<int32_t>(1280 * scale);
        auto height = static_cast<int32_t>(820 * scale);
        if (width > work.Width * 0.92 || height > work.Height * 0.92)
        {
            if (auto presenter = appWindow.Presenter().try_as<Microsoft::UI::Windowing::OverlappedPresenter>())
            {
                presenter.Maximize();
                return;
            }
            width = (std::min)(width, work.Width);
            height = (std::min)(height, work.Height);
        }
        appWindow.MoveAndResize({ work.X + (work.Width - width) / 2, work.Y + (work.Height - height) / 2, width, height });
    }

    // ---- GPU footer: full telemetry in an open pane, one icon in an icon-only rail -------------------------------
    void MainWindow::ShowFooter(bool full)
    {
        FooterFull().Visibility(full ? Visibility::Visible : Visibility::Collapsed);
        FooterCompact().Visibility(full ? Visibility::Collapsed : Visibility::Visible);
    }

    void MainWindow::Nav_DisplayModeChanged(NavigationView const& sender, NavigationViewDisplayModeChangedEventArgs const&)
    {
        bool minimal = sender.DisplayMode() == NavigationViewDisplayMode::Minimal;
        ShowFooter(sender.IsPaneOpen() && !minimal);
        // In the narrow (menu button) mode the button floats over the top-left of the content, where every
        // page puts its title; the pages start one button-height lower so the two never overlap.
        ContentFrame().Margin(minimal ? ThicknessHelper::FromLengths(0, 40, 0, 0) : ThicknessHelper::FromUniformLength(0));
    }

    // a narrow title bar drops the "local AI workstation" subtitle so the project picker keeps its room
    void MainWindow::AppTitleBar_SizeChanged(IInspectable const&, SizeChangedEventArgs const& args)
    {
        TitleSubtitle().Visibility(args.NewSize().Width < 760 ? Visibility::Collapsed : Visibility::Visible);
    }

    void MainWindow::Nav_PaneOpening(NavigationView const&, IInspectable const&)
    {
        ShowFooter(true);
    }

    void MainWindow::Nav_PaneClosing(NavigationView const&, NavigationViewPaneClosingEventArgs const&)
    {
        ShowFooter(false);
    }

    // ---- the app-wide refresh loop ------------------------------------------------------------------------
    fire_and_forget MainWindow::RefreshTick()
    {
        if (m_refreshing) co_return;   // the previous refresh has not finished yet
        m_refreshing = true;
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();

        co_await aiwf::EngineSupervisor::Instance().RefreshAsync();
        co_await resume_background();
        auto sample = aiwf::GpuMonitor::Instance().Sample();
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile

        // footer: card name and memory in use, always visible
        if (sample.available)
        {
            std::wstring name = sample.name;
            for (wchar_t const* prefix : { L"NVIDIA ", L"GeForce " })
            {
                if (name.rfind(prefix, 0) == 0) name.erase(0, wcslen(prefix));
            }
            wchar_t text[160]{};
            FooterGpu().Text(name);
            double memoryPercent = std::clamp(sample.totalBytes ? 100.0 * sample.usedBytes / sample.totalBytes : 0.0, 0.0, 100.0);
            double loadPercent = std::clamp(static_cast<double>(sample.gpuUtilPercent), 0.0, 100.0);
            swprintf_s(text, L"Video memory: %.1f of %.1f GB (%.0f%%)", sample.usedBytes / 1073741824.0,
                       sample.totalBytes / 1073741824.0, memoryPercent);
            FooterVramLabel().Text(text);
            FooterVram().Value(memoryPercent);
            swprintf_s(text, L"GPU load: %.0f%%", loadPercent);
            FooterLoadLabel().Text(text);
            FooterLoad().Value(loadPercent);

            // the icon rail shows memory use as a short number and the full lines in its tooltip
            swprintf_s(text, L"%.0f%%", memoryPercent);
            FooterCompactText().Text(text);
            wchar_t tip[200]{};
            swprintf_s(tip, L"%s\nVideo memory %.1f of %.1f GB (%.0f%%)\nGPU load %.0f%%", name.c_str(), sample.usedBytes / 1073741824.0,
                       sample.totalBytes / 1073741824.0, memoryPercent, loadPercent);
            ToolTipService::SetToolTip(FooterCompact(), box_value(hstring(tip)));
        }
        else
        {
            FooterGpu().Text(hstring(sample.error.empty() ? L"No GPU telemetry" : sample.error));
            FooterVram().Value(0);
            FooterLoad().Value(0);
            FooterVramLabel().Text(L"Video memory: unavailable");
            FooterLoadLabel().Text(L"GPU load: unavailable");
            FooterCompactText().Text(L"--");
            ToolTipService::SetToolTip(FooterCompact(), box_value(L"GPU telemetry unavailable"));
        }

        // engines start themselves: the engine API is light (no GPU) and every data page needs it
        auto& supervisor = aiwf::EngineSupervisor::Instance();
        auto api = supervisor.Status(L"engine-api").state;
        if (!m_engineApiStartTried && api == aiwf::EngineState::Stopped && supervisor.Find(L"engine-api"))
        {
            m_engineApiStartTried = true;
            supervisor.StartAsync(L"engine-api");
        }
        bool pickerOpen = ProjectPicker().IsDropDownOpen();
        if (!pickerOpen && m_projectPickerSyncPending)
            SyncProjectPickerToState();
        if ((!m_projectsLoaded || m_projectPickerRefreshPending) && supervisor.IsAnswering(L"engine-api") && !pickerOpen)
        {
            LoadProjectsAsync();
        }
        else if (!m_projectsLoaded && !pickerOpen)
        {
            // the saved project is still in focus; say so instead of "Choose a project" while the list is unreachable
            auto saved = aiwf::AppState::Get().ProjectName();
            ProjectPicker().PlaceholderText(saved.empty() ? hstring(L"Choose a project") : hstring(saved + L" (waiting for the engine API)"));
        }
        m_refreshing = false;
    }

    // ---- project picker ----------------------------------------------------------------------------------------
    fire_and_forget MainWindow::LoadProjectsAsync()
    {
        if (m_loadingProjects) co_return;
        if (ProjectPicker().IsDropDownOpen())
        {
            m_projectPickerRefreshPending = true;
            aiwf::diagnostics::MarkProjectSwitchStep("main.projects-load-deferred-open");
            co_return;
        }
        m_loadingProjects = true;
        m_projectPickerRefreshPending = true;
        aiwf::diagnostics::MarkProjectSwitchStep("main.projects-load-start");
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        JsonObject body{ nullptr };
        try
        {
            body = co_await aiwf::http::GetJsonAsync(aiwf::AppState::Get().EngineApi() + L"/projects");
        }
        catch (hresult_error const&)
        {
            // the engine API is not up yet; the refresh loop tries again
        }
        co_await wil::resume_foreground(dispatcher);
        m_loadingProjects = false;
        if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile
        if (ProjectPicker().IsDropDownOpen())
        {
            m_projectPickerRefreshPending = true;
            aiwf::diagnostics::MarkProjectSwitchStep("main.projects-load-deferred-response");
            co_return;
        }
        if (!body) co_return;

        try
        {
            auto current = aiwf::AppState::Get().ProjectId();
            auto previousSuppress = m_suppressPicker;
            m_suppressPicker = true;
            auto restoreSuppress = wil::scope_exit([this, previousSuppress] { m_suppressPicker = previousSuppress; });
            aiwf::diagnostics::MarkProjectSwitchStep("main.projects-list-clear");
            ProjectPicker().PlaceholderText(L"Choose a project");
            auto items = ProjectPicker().Items();
            items.Clear();
            // this loop fills the picker and re-selects the project in focus
            for (auto const& value : aiwf::http::Arr(body, L"projects"))
            {
                auto project = value.GetObject();
                ComboBoxItem item;
                item.Content(box_value(aiwf::http::Str(project, L"name")));
                item.Tag(box_value(aiwf::http::Str(project, L"project_id")));
                items.Append(item);
                if (aiwf::http::Str(project, L"project_id") == current) ProjectPicker().SelectedItem(item);
            }
            m_projectsLoaded = true;
            m_projectPickerRefreshPending = false;
            aiwf::diagnostics::MarkProjectSwitchStep("main.projects-list-complete");
        }
        catch (hresult_error const& error)
        {
            aiwf::diagnostics::LogProjectSwitchFailure("main.projects-list-mutation", error);
            throw;
        }
    }

    void MainWindow::SyncProjectPickerToState()
    {
        try
        {
            if (ProjectPicker().IsDropDownOpen())
            {
                auto action = aiwf::ui_policy::DecideProjectPickerAction(true, false, false, false);
                if (action == aiwf::ui_policy::ProjectPickerAction::DeferUntilClosed)
                {
                    m_projectPickerSyncPending = true;
                    aiwf::diagnostics::MarkProjectSwitchStep("main.project-picker-sync-deferred-open");
                    return;
                }
            }

            auto& state = aiwf::AppState::Get();
            auto projectId = state.ProjectId();
            auto projectName = state.ProjectName();
            ComboBoxItem currentItem{ nullptr };
            aiwf::diagnostics::MarkProjectSwitchStep("main.project-picker-sync-read-items");
            for (auto const& value : ProjectPicker().Items())
            {
                auto item = value.try_as<ComboBoxItem>();
                if (item && std::wstring(unbox_value_or<hstring>(item.Tag(), L"")) == projectId)
                {
                    currentItem = item;
                    break;
                }
            }

            auto selectedItem = ProjectPicker().SelectedItem().try_as<ComboBoxItem>();
            bool selectionMatches = selectedItem &&
                std::wstring(unbox_value_or<hstring>(selectedItem.Tag(), L"")) == projectId;
            auto action = aiwf::ui_policy::DecideProjectPickerAction(false, !projectId.empty(),
                                                                      static_cast<bool>(currentItem), selectionMatches);
            m_projectPickerRefreshPending = aiwf::ui_policy::NeedsProjectRefreshAfterSync(
                action, m_projectPickerRefreshPending, m_projectsLoaded);
            aiwf::diagnostics::MarkProjectSwitchStep("main.project-picker-sync");
            switch (action)
            {
            case aiwf::ui_policy::ProjectPickerAction::DeferUntilClosed:
                m_projectPickerSyncPending = true;
                aiwf::diagnostics::MarkProjectSwitchStep("main.project-picker-sync-deferred-open");
                return;
            case aiwf::ui_policy::ProjectPickerAction::ClearSelection:
            {
                m_projectPickerSyncPending = false;
                if (ProjectPicker().SelectedIndex() != -1)
                {
                    auto previousSuppress = m_suppressPicker;
                    m_suppressPicker = true;
                    auto restoreSuppress = wil::scope_exit([this, previousSuppress] { m_suppressPicker = previousSuppress; });
                    ProjectPicker().SelectedIndex(-1);
                }
                return;
            }
            case aiwf::ui_policy::ProjectPickerAction::RefreshUnknownProject:
                m_projectPickerSyncPending = false;
                aiwf::diagnostics::MarkProjectSwitchStep("main.project-picker-sync-unknown");
                if (!m_loadingProjects && aiwf::EngineSupervisor::Instance().IsAnswering(L"engine-api")) LoadProjectsAsync();
                return;
            case aiwf::ui_policy::ProjectPickerAction::KeepSelection:
                m_projectPickerSyncPending = false;
                if (!projectName.empty() && unbox_value_or<hstring>(currentItem.Content(), L"") != hstring(projectName))
                    currentItem.Content(box_value(hstring(projectName)));
                aiwf::diagnostics::MarkProjectSwitchStep("main.project-picker-sync-kept");
                return;
            case aiwf::ui_policy::ProjectPickerAction::SelectKnownProject:
            {
                m_projectPickerSyncPending = false;
                auto previousSuppress = m_suppressPicker;
                m_suppressPicker = true;
                auto restoreSuppress = wil::scope_exit([this, previousSuppress] { m_suppressPicker = previousSuppress; });
                if (!projectName.empty() && unbox_value_or<hstring>(currentItem.Content(), L"") != hstring(projectName))
                    currentItem.Content(box_value(hstring(projectName)));
                ProjectPicker().SelectedItem(currentItem);
                aiwf::diagnostics::MarkProjectSwitchStep("main.project-picker-sync-selected-known");
                return;
            }
            }
        }
        catch (hresult_error const& error)
        {
            aiwf::diagnostics::LogProjectSwitchFailure("main.project-picker-sync", error);
            throw;
        }
    }

    void MainWindow::ProjectPicker_DropDownClosed(IInspectable const&, IInspectable const&)
    {
        aiwf::diagnostics::MarkProjectSwitchStep("main.project-picker-drop-down-closed");
        auto weak = get_weak();
        DispatcherQueue().TryEnqueue([weak]
        {
            if (auto self = weak.get(); self && !aiwf::AppState::Get().ShuttingDown() && !self->ProjectPicker().IsDropDownOpen())
            {
                if (self->m_projectPickerSyncPending || self->m_projectPickerRefreshPending)
                    self->SyncProjectPickerToState();
                if ((!self->m_projectsLoaded || self->m_projectPickerRefreshPending) &&
                    !self->m_loadingProjects && aiwf::EngineSupervisor::Instance().IsAnswering(L"engine-api"))
                    self->LoadProjectsAsync();
            }
        });
    }

    void MainWindow::ProjectPicker_SelectionChanged(IInspectable const&, SelectionChangedEventArgs const&)
    {
        if (m_suppressPicker) return;
        if (auto item = ProjectPicker().SelectedItem().try_as<ComboBoxItem>())
        {
            auto& state = aiwf::AppState::Get();
            if (!state.SetProject(std::wstring(unbox_value_or<hstring>(item.Tag(), L"")),
                                  std::wstring(unbox_value_or<hstring>(item.Content(), L""))))
            {
                auto previousId = state.ProjectId();
                if (ProjectPicker().IsDropDownOpen())
                {
                    m_projectPickerSyncPending = true;
                }
                else
                {
                    auto previousSuppress = m_suppressPicker;
                    m_suppressPicker = true;
                    auto restoreSuppress = wil::scope_exit([this, previousSuppress] { m_suppressPicker = previousSuppress; });
                    for (auto const& value : ProjectPicker().Items())
                    {
                        auto previous = value.try_as<ComboBoxItem>();
                        if (previous && std::wstring(unbox_value_or<hstring>(previous.Tag(), L"")) == previousId)
                        {
                            ProjectPicker().SelectedItem(previous);
                            break;
                        }
                    }
                }
                ui::ShowMessageAsync(ProjectPicker().XamlRoot(), L"Project choice was not saved",
                                     L"AES Studio could not write %LOCALAPPDATA%\\Programs\\AES Studio Data\\settings.json. The previous project remains in focus; check that the settings file and folder are writable, then try again.");
            }
        }
    }

    // ---- navigation ---------------------------------------------------------------------------------------------
    void MainWindow::Nav_SelectionChanged(NavigationView const&, NavigationViewSelectionChangedEventArgs const& args)
    {
        if (args.IsSettingsSelected())
        {
            NavigateTo(L"settings");
            return;
        }
        if (auto item = args.SelectedItem().try_as<NavigationViewItem>())
        {
            NavigateTo(unbox_value_or<hstring>(item.Tag(), L"home"));
        }
    }

    void MainWindow::NavigateTo(hstring const& tag)
    {
        // one page per task; pages that hold work in progress keep their state between visits
        if (tag == L"chat") ContentFrame().Navigate(xaml_typename<AiwfDesktop::ChatPage>());
        else if (tag == L"create") ContentFrame().Navigate(xaml_typename<AiwfDesktop::CreatePage>());
        else if (tag == L"datasets") ContentFrame().Navigate(xaml_typename<AiwfDesktop::DatasetsPage>());
        else if (tag == L"train") ContentFrame().Navigate(xaml_typename<AiwfDesktop::TrainPage>());
        else if (tag == L"projects") ContentFrame().Navigate(xaml_typename<AiwfDesktop::ProjectsPage>());
        else if (tag == L"settings") ContentFrame().Navigate(xaml_typename<AiwfDesktop::SettingsPage>());
        else if (tag == L"setup") ContentFrame().Navigate(xaml_typename<AiwfDesktop::SetupPage>());
        else ContentFrame().Navigate(xaml_typename<AiwfDesktop::HomePage>());
    }

    fire_and_forget MainWindow::ShowStartupProblemAsync(hstring message)
    {
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        // wait until the window has content (and so a XamlRoot) before showing a dialog
        co_await resume_after(std::chrono::milliseconds(800));
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile
        if (auto root = Content().XamlRoot())
        {
            co_await aiwf::ui::ShowMessageAsync(root, L"Engines are not configured", message);
        }
    }

    // ---- closing: nothing this app started keeps running -----------------------------------------------------
    // Order matters: first tell every timer, callback and pending async continuation to stop touching
    // the UI, then drop the callbacks, and only then end the engines this app started. This keeps queued
    // refresh work away from torn-down XAML; the App also drops its window reference on close
    // (App.xaml.cpp) so the cached pages are destroyed while XAML is still alive. The Mica backdrop is
    // left for the window to release itself: clearing it here made its controller raise a WinRT error
    // while unhooking a composition batch that was already gone.
    void MainWindow::OnClosed(IInspectable const&, WindowEventArgs const&)
    {
        aiwf::AppState::Get().BeginShutdown();
        aiwf::EngineSupervisor::Instance().ClearHandlers();
        if (m_timer) m_timer.Stop();
        aiwf::EngineSupervisor::Instance().StopAll();
    }
}
