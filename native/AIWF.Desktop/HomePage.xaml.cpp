// Home page implementation: GPU card, engine rows, current project.
#include "pch.h"
#include "HomePage.xaml.h"
#if __has_include("HomePage.g.cpp")
#include "HomePage.g.cpp"
#endif
#include "Services/AppState.h"
#include "Services/EngineSupervisor.h"
#include "Services/Http.h"
#include "Services/Paths.h"
#include "Services/Ui.h"

using namespace winrt;
using namespace winrt::Windows::Foundation;
using namespace winrt::Windows::Data::Json;
using namespace winrt::Microsoft::UI::Xaml;
using namespace winrt::Microsoft::UI::Xaml::Controls;
using namespace winrt::Microsoft::UI::Xaml::Navigation;
namespace ui = aiwf::ui;

namespace winrt::AiwfDesktop::implementation
{
    namespace
    {
        // this is the status-light color section (green running, blue external, amber starting, red problem)
        constexpr uint32_t ColorRunning = 0xFF2DA44E;
        constexpr uint32_t ColorExternal = 0xFF3A96DD;
        constexpr uint32_t ColorStarting = 0xFFE3A21A;
        constexpr uint32_t ColorOff = 0xFF8A8A8A;
        constexpr uint32_t ColorProblem = 0xFFD13438;
        constexpr uint32_t ColorOtherApps = 0xFF9A9A9A;

        uint32_t StateColor(aiwf::EngineState state)
        {
            switch (state)
            {
            case aiwf::EngineState::Running: return ColorRunning;
            case aiwf::EngineState::RunningExternal: return ColorExternal;
            case aiwf::EngineState::Starting: return ColorStarting;
            case aiwf::EngineState::Failed: return ColorProblem;
            default: return ColorOff;
            }
        }

        std::wstring Gigabytes(uint64_t bytes)
        {
            wchar_t text[32]{};
            swprintf_s(text, L"%.1f", bytes / (1024.0 * 1024 * 1024));
            return text;
        }

        // VRAM held by one engine: the sum over the processes the supervisor counts for it
        uint64_t EngineVram(aiwf::EngineStatus const& status, aiwf::GpuSample const& sample)
        {
            uint64_t bytes = 0;
            for (auto pid : status.pids)
            {
                auto it = sample.processVram.find(pid);
                if (it != sample.processVram.end()) bytes += it->second;
            }
            return bytes;
        }

        // what each ledger event kind means to a person
        std::wstring KindLabel(std::wstring const& kind)
        {
            if (kind == L"image_generated") return L"Image jobs finished";
            if (kind == L"dataset_catalog") return L"Catalog additions";
            if (kind == L"retrain_import") return L"Packages imported for training";
            if (kind == L"retrain_preflight") return L"Training plans checked";
            if (kind == L"qwen_context_sent") return L"Chat questions with project context";
            return kind;
        }
    }

    HomePage::HomePage()
    {
        InitializeComponent();
        BuildEngineRows();
        // telemetry tiles and the memory legend reflow when the page width changes
        auto weak = get_weak();
        TelemetryGrid().SizeChanged([weak](auto&&, SizeChangedEventArgs const& args)
        {
            if (auto self = weak.get(); self && !aiwf::AppState::Get().ShuttingDown()) self->ArrangeTiles(args.NewSize().Width);
        });
        VramLegend().SizeChanged([weak](auto&&, SizeChangedEventArgs const& args)
        {
            if (auto self = weak.get(); self && !aiwf::AppState::Get().ShuttingDown()) self->ArrangeLegend(args.NewSize().Width);
        });
    }

    // ---- responsive arrangement -----------------------------------------------------------------------------
    // four tiles in one row when there is room for them, otherwise two rows of two. The widest value,
    // "13.5 / 16.0 GB" in the large number style, needs about 185 units, so four across start at 760.
    void HomePage::ArrangeTiles(double width)
    {
        int columns = width >= 760 ? 4 : 2;
        if (columns == m_tileColumns) return;
        m_tileColumns = columns;
        auto grid = TelemetryGrid();
        grid.ColumnDefinitions().Clear();
        grid.RowDefinitions().Clear();
        for (int i = 0; i < columns; ++i)
        {
            ColumnDefinition column;
            column.Width(GridLengthHelper::FromValueAndType(1, GridUnitType::Star));
            grid.ColumnDefinitions().Append(column);
        }
        auto tiles = grid.Children();
        int rows = (static_cast<int>(tiles.Size()) + columns - 1) / columns;
        for (int i = 0; i < rows; ++i) grid.RowDefinitions().Append(RowDefinition());
        for (uint32_t i = 0; i < tiles.Size(); ++i)
        {
            auto tile = tiles.GetAt(i).as<FrameworkElement>();
            Grid::SetColumn(tile, static_cast<int32_t>(i) % columns);
            Grid::SetRow(tile, static_cast<int32_t>(i) / columns);
        }
    }

    // legend entries flow left to right in as many 220-unit columns as the width allows
    void HomePage::ArrangeLegend(double width)
    {
        int columns = (std::max)(1, static_cast<int>((width + 18) / (220 + 18)));
        auto grid = VramLegend();
        grid.Children().Clear();
        grid.ColumnDefinitions().Clear();
        grid.RowDefinitions().Clear();
        for (int i = 0; i < columns; ++i)
        {
            ColumnDefinition column;
            column.Width(GridLengthHelper::FromValueAndType(1, GridUnitType::Star));
            grid.ColumnDefinitions().Append(column);
        }
        int rows = (static_cast<int>(m_legendItems.size()) + columns - 1) / columns;
        for (int i = 0; i < rows; ++i) grid.RowDefinitions().Append(RowDefinition());
        for (size_t i = 0; i < m_legendItems.size(); ++i)
        {
            Grid::SetColumn(m_legendItems[i], static_cast<int32_t>(i) % columns);
            Grid::SetRow(m_legendItems[i], static_cast<int32_t>(i) / columns);
            grid.Children().Append(m_legendItems[i]);
        }
    }

    void HomePage::StartApi_Click(IInspectable const&, RoutedEventArgs const&)
    {
        ui::StartEnginesAsync({ L"engine-api" });
    }

    // ---- page lifetime: live updates only while Home is on screen ------------------------------------
    void HomePage::OnNavigatedTo(NavigationEventArgs const&)
    {
        auto weak = get_weak();
        auto dispatcher = DispatcherQueue();
        m_statusToken = aiwf::EngineSupervisor::Instance().StatusChanged([weak, dispatcher]
        {
            dispatcher.TryEnqueue([weak]
            {
                if (auto self = weak.get(); self && !aiwf::AppState::Get().ShuttingDown()) self->UpdateEngineRows();
            });
        });
        m_projectToken = aiwf::AppState::Get().ProjectChanged([weak]
        {
            if (auto self = weak.get(); self && !aiwf::AppState::Get().ShuttingDown()) self->LoadProjectAsync();
        });
        m_timer = dispatcher.CreateTimer();
        m_timer.Interval(std::chrono::seconds(2));
        m_timer.Tick([weak](auto&&, auto&&)
        {
            if (auto self = weak.get(); self && !aiwf::AppState::Get().ShuttingDown()) self->RefreshAsync();
        });
        m_timer.Start();
        UpdateEngineRows();
        RefreshAsync();
        LoadProjectAsync();
    }

    void HomePage::OnNavigatedFrom(NavigationEventArgs const&)
    {
        ++m_projectGeneration;
        if (m_timer) m_timer.Stop();
        aiwf::EngineSupervisor::Instance().StatusChanged(m_statusToken);
        aiwf::AppState::Get().ProjectChanged(m_projectToken);
    }

    // ---- GPU card ---------------------------------------------------------------------------------------
    fire_and_forget HomePage::RefreshAsync()
    {
        if (m_refreshing) co_return;   // the previous sample is still being taken
        m_refreshing = true;
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        co_await resume_background();
        auto sample = aiwf::GpuMonitor::Instance().Sample();
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile
        m_lastSample = std::move(sample);
        ShowGpu(m_lastSample);
        UpdateEngineRows();
        m_refreshing = false;
    }

    void HomePage::ShowGpu(aiwf::GpuSample const& sample)
    {
        if (!sample.available)
        {
            GpuName().Text(L"No GPU telemetry");
            GpuDriver().Text(sample.error);
            GpuNote().Text(L"");
            return;
        }
        GpuName().Text(sample.name);
        std::wstring facts = L"Driver " + sample.driver;
        if (!sample.cuda.empty()) facts += L"  ·  CUDA " + sample.cuda;
        facts += L"  ·  " + Gigabytes(sample.totalBytes) + L" GB video memory";
        GpuDriver().Text(facts);

        // this block fills the four headline numbers
        uint64_t free = sample.totalBytes > sample.usedBytes ? sample.totalBytes - sample.usedBytes : 0;
        VramValue().Text(Gigabytes(sample.usedBytes) + L" / " + Gigabytes(sample.totalBytes) + L" GB");
        VramFree().Text(aiwf::paths::FormatBytes(free) + L" free");
        UtilValue().Text(std::to_wstring(sample.gpuUtilPercent) + L" %");
        ClockValue().Text(sample.graphicsClockMHz ? std::to_wstring(sample.graphicsClockMHz) + L" MHz" : L"");
        TempValue().Text(sample.temperatureC ? std::to_wstring(sample.temperatureC) + L" °C" : L"-");
        wchar_t power[48]{};
        swprintf_s(power, L"%.0f W", sample.powerWatts);
        PowerValue().Text(power);
        if (sample.powerLimitWatts > 0)
        {
            swprintf_s(power, L"of %.0f W limit", sample.powerLimitWatts);
            PowerLimit().Text(power);
        }

        // this block splits used memory into one segment per engine, then everything else
        struct Segment
        {
            std::wstring label;
            uint64_t bytes;
            uint32_t color;
        };
        std::vector<Segment> segments;
        uint64_t engineTotal = 0;
        auto& supervisor = aiwf::EngineSupervisor::Instance();
        for (auto const& engine : supervisor.Engines())
        {
            auto bytes = EngineVram(supervisor.Status(engine.id), sample);
            if (bytes == 0) continue;
            segments.push_back({ engine.name, bytes, engine.color });
            engineTotal += bytes;
        }
        uint64_t other = sample.usedBytes > engineTotal ? sample.usedBytes - engineTotal : 0;
        if (other > 0) segments.push_back({ L"Windows and other apps", other, ColorOtherApps });

        auto bar = VramBar();
        bar.ColumnDefinitions().Clear();
        bar.Children().Clear();
        m_legendItems.clear();
        int column = 0;
        // this loop draws each segment as a star-sized column, so widths are proportional to bytes
        for (auto const& segment : segments)
        {
            ColumnDefinition definition;
            definition.Width(GridLengthHelper::FromValueAndType(static_cast<double>(segment.bytes), GridUnitType::Star));
            bar.ColumnDefinitions().Append(definition);
            Shapes::Rectangle block;
            block.Fill(ui::ColorBrush(segment.color));
            Grid::SetColumn(block, column++);
            ToolTipService::SetToolTip(block, box_value(hstring(segment.label + L": " + aiwf::paths::FormatBytes(segment.bytes))));
            bar.Children().Append(block);

            StackPanel item;
            item.Orientation(Orientation::Horizontal);
            item.Spacing(6);
            Shapes::Ellipse dot;
            dot.Width(10);
            dot.Height(10);
            dot.Fill(ui::ColorBrush(segment.color));
            dot.VerticalAlignment(VerticalAlignment::Center);
            item.Children().Append(dot);
            auto label = ui::Text(hstring(segment.label + L"  " + aiwf::paths::FormatBytes(segment.bytes)), L"CaptionTextBlockStyle");
            label.TextWrapping(TextWrapping::NoWrap);
            label.TextTrimming(TextTrimming::CharacterEllipsis);
            item.Children().Append(label);
            m_legendItems.push_back(item);
        }
        if (free > 0)
        {
            ColumnDefinition definition;
            definition.Width(GridLengthHelper::FromValueAndType(static_cast<double>(free), GridUnitType::Star));
            bar.ColumnDefinitions().Append(definition);
            auto freeLabel = ui::Text(hstring(L"Free  " + aiwf::paths::FormatBytes(free)), L"CaptionTextBlockStyle", 0.7);
            freeLabel.TextWrapping(TextWrapping::NoWrap);
            m_legendItems.push_back(freeLabel);
        }
        ArrangeLegend(VramLegend().ActualWidth());
        GpuNote().Text(L"From the NVIDIA driver (NVML) and Windows GPU memory counters, refreshed every 2 seconds. "
                       L"An engine's share counts every process it runs on this GPU.");
    }

    // ---- engine rows -------------------------------------------------------------------------------------
    void HomePage::BuildEngineRows()
    {
        EngineList().Children().Clear();
        m_rows.clear();
        auto weak = get_weak();
        for (auto const& engine : aiwf::EngineSupervisor::Instance().Engines())
        {
            // status dot | name, role and detail (takes the spare width) | state over VRAM | action button.
            // No fixed pixel columns, so a narrow window shrinks the text column instead of clipping.
            Grid grid;
            grid.ColumnSpacing(16);
            for (auto width : { GridLengthHelper::Auto(), GridLengthHelper::FromValueAndType(1, GridUnitType::Star),
                                GridLengthHelper::Auto(), GridLengthHelper::Auto() })
            {
                ColumnDefinition definition;
                definition.Width(width);
                grid.ColumnDefinitions().Append(definition);
            }

            EngineRow row;
            row.dot = Shapes::Ellipse();
            row.dot.Width(12);
            row.dot.Height(12);
            row.dot.VerticalAlignment(VerticalAlignment::Center);
            grid.Children().Append(row.dot);

            StackPanel text;
            text.Spacing(1);
            text.Children().Append(ui::Text(hstring(engine.name), L"BodyStrongTextBlockStyle"));
            text.Children().Append(ui::Text(hstring(engine.role), L"CaptionTextBlockStyle", 0.75));
            row.detail = ui::Text(L"", L"CaptionTextBlockStyle", 0.6);
            text.Children().Append(row.detail);
            Grid::SetColumn(text, 1);
            grid.Children().Append(text);

            StackPanel status;
            status.VerticalAlignment(VerticalAlignment::Center);
            row.state = ui::Text(L"", L"BodyStrongTextBlockStyle");
            row.state.TextWrapping(TextWrapping::NoWrap);
            row.state.HorizontalAlignment(HorizontalAlignment::Right);
            status.Children().Append(row.state);
            row.vram = ui::Text(L"", L"CaptionTextBlockStyle", 0.8);
            row.vram.TextWrapping(TextWrapping::NoWrap);
            row.vram.HorizontalAlignment(HorizontalAlignment::Right);
            status.Children().Append(row.vram);
            Grid::SetColumn(status, 2);
            grid.Children().Append(status);

            row.action = Button();
            row.action.MinWidth(96);
            row.action.VerticalAlignment(VerticalAlignment::Center);
            row.action.Click([weak, id = engine.id](auto&&, auto&&)
            {
                if (auto self = weak.get(); self && !aiwf::AppState::Get().ShuttingDown()) self->EngineActionAsync(id);
            });
            Automation::AutomationProperties::SetName(row.action, hstring(L"Start or stop " + engine.name));
            Grid::SetColumn(row.action, 3);
            grid.Children().Append(row.action);

            auto card = ui::Card(grid);
            card.Padding(ThicknessHelper::FromLengths(16, 12, 16, 12));
            EngineList().Children().Append(card);
            m_rows.emplace(engine.id, row);
        }
    }

    void HomePage::UpdateEngineRows()
    {
        auto& supervisor = aiwf::EngineSupervisor::Instance();
        bool anyOwned = false;
        for (auto& [id, row] : m_rows)
        {
            auto status = supervisor.Status(id);
            row.dot.Fill(ui::ColorBrush(StateColor(status.state)));
            row.state.Text(aiwf::StateLabel(status.state));
            row.detail.Text(status.detail);
            auto bytes = EngineVram(status, m_lastSample);
            row.vram.Text(bytes > 0 ? hstring(aiwf::paths::FormatBytes(bytes) + L" VRAM") : hstring());

            // the button offers only what this app may do in this state
            switch (status.state)
            {
            case aiwf::EngineState::Running:
                anyOwned = true;
                row.action.Content(box_value(L"Stop"));
                row.action.IsEnabled(true);
                break;
            case aiwf::EngineState::RunningExternal:
                row.action.Content(box_value(L"In use"));
                row.action.IsEnabled(false);
                ToolTipService::SetToolTip(row.action, box_value(L"Started outside AIWF Studio, so it is left as is."));
                break;
            case aiwf::EngineState::Starting:
                anyOwned = true;
                row.action.Content(box_value(L"Starting..."));
                row.action.IsEnabled(false);
                break;
            default:
            {
                // an engine the engine list does not know how to start gets no Start that could only fail
                auto spec = supervisor.Find(id);
                bool startable = spec && !spec->processes.empty();
                row.action.Content(box_value(startable ? L"Start" : L"Not set up"));
                row.action.IsEnabled(startable);
                if (!startable)
                {
                    ToolTipService::SetToolTip(row.action, box_value(L"The engine list has no start command for this engine. "
                                                                     L"Start it with its own launcher, or add its process in Configure Studio > Open engine list."));
                }
                break;
            }
            }
        }
        StopAllButton().IsEnabled(anyOwned || supervisor.HasPendingStarts());
        ui::UpdateEngineBar(ApiBar(), { L"engine-api" });
    }

    fire_and_forget HomePage::EngineActionAsync(std::wstring id)
    {
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        auto& supervisor = aiwf::EngineSupervisor::Instance();
        if (supervisor.Status(id).state == aiwf::EngineState::Running)
        {
            std::wstring message;
            supervisor.Stop(id, message);
            UpdateEngineRows();
            co_return;
        }
        co_await supervisor.StartAsync(id);
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile
        auto status = supervisor.Status(id);
        if (status.state == aiwf::EngineState::Failed)
        {
            auto spec = supervisor.Find(id);
            EngineMessage().Title(hstring((spec ? spec->name : id) + L" did not start"));
            EngineMessage().Message(hstring(status.detail));
            EngineMessage().Severity(InfoBarSeverity::Error);
            EngineMessage().IsOpen(true);
        }
        UpdateEngineRows();
    }

    void HomePage::StartAll_Click(IInspectable const&, RoutedEventArgs const&)
    {
        std::vector<std::wstring> ids;
        for (auto const& engine : aiwf::EngineSupervisor::Instance().Engines()) ids.push_back(engine.id);
        EngineMessage().IsOpen(false);
        [](std::vector<std::wstring> all) -> fire_and_forget { co_await ui::StartEnginesAsync(std::move(all)); }(std::move(ids));
    }

    void HomePage::StopAll_Click(IInspectable const&, RoutedEventArgs const&)
    {
        aiwf::EngineSupervisor::Instance().StopAll();
        aiwf::EngineSupervisor::Instance().RefreshAsync();
        UpdateEngineRows();
    }

    // ---- the project in focus -------------------------------------------------------------------------
    fire_and_forget HomePage::LoadProjectAsync()
    {
        auto generation = ++m_projectGeneration;
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        auto& state = aiwf::AppState::Get();
        auto id = state.ProjectId();
        auto panel = ProjectSummary();
        panel.Children().Clear();
        if (id.empty())
        {
            panel.Children().Append(ui::Text(L"No project chosen yet. Pick one in the title bar, or create one on the Projects page. "
                                             L"Images, datasets, training plans and chat context are recorded against it."));
            co_return;
        }
        panel.Children().Append(ui::Text(hstring(state.ProjectName()), L"SubtitleTextBlockStyle"));
        auto idText = ui::Text(hstring(id), L"CaptionTextBlockStyle", 0.6);
        idText.IsTextSelectionEnabled(true);
        panel.Children().Append(idText);
        // the error is kept and shown after returning to the UI thread (no co_await is allowed in a catch)
        JsonObject body{ nullptr };
        hstring failure;
        try
        {
            body = co_await aiwf::http::GetJsonAsync(state.EngineApi() + L"/projects/" + aiwf::http::EscapeSegment(hstring(id)));
        }
        catch (hresult_error const& error)
        {
            failure = error.message();
        }
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile
        if (generation != m_projectGeneration || id != state.ProjectId()) co_return;
        if (!failure.empty())
        {
            panel.Children().Append(ui::Text(L"Project details are not available: " + failure, L"CaptionTextBlockStyle", 0.7));
            co_return;
        }
        auto project = aiwf::http::Obj(body, L"project");
        auto counts = aiwf::http::Obj(project, L"event_counts");
        if (!counts || counts.Size() == 0)
        {
            panel.Children().Append(ui::Text(L"Nothing recorded yet.", L"BodyTextBlockStyle", 0.75));
            co_return;
        }
        // this loop lists one line per kind of recorded work
        for (auto const& pair : counts)
        {
            auto count = static_cast<int>(pair.Value().GetNumber());
            panel.Children().Append(ui::Text(hstring(KindLabel(std::wstring(pair.Key())) + L": " + std::to_wstring(count))));
        }
        auto events = aiwf::http::Arr(project, L"events");
        if (events.Size() > 0)
        {
            auto last = events.GetObjectAt(events.Size() - 1);
            panel.Children().Append(ui::Text(L"Last activity " + ui::FriendlyTime(aiwf::http::Str(last, L"at")), L"CaptionTextBlockStyle", 0.6));
        }
    }
}
