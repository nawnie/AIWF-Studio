// Create page implementation: image jobs through the engine API, results read from disk.
#include "pch.h"
#include "CreatePage.xaml.h"
#if __has_include("CreatePage.g.cpp")
#include "CreatePage.g.cpp"
#endif
#include "Services/AppState.h"
#include "Services/EngineSupervisor.h"
#include "Services/Http.h"
#include "Services/Ui.h"

using namespace winrt;
using namespace winrt::Windows::Foundation;
using namespace winrt::Windows::Data::Json;
using namespace winrt::Windows::Storage;
using namespace winrt::Microsoft::UI::Xaml;
using namespace winrt::Microsoft::UI::Xaml::Controls;
using namespace winrt::Microsoft::UI::Xaml::Media::Imaging;
using namespace winrt::Microsoft::UI::Xaml::Navigation;
namespace ui = aiwf::ui;
namespace http = aiwf::http;

namespace winrt::AiwfDesktop::implementation
{
    namespace
    {
        // Studio's output folder + a relative path from the engine API ("qwen-image/2026-10-07/x.png")
        std::wstring JoinOutput(std::wstring const& root, std::wstring relative)
        {
            std::replace(relative.begin(), relative.end(), L'/', L'\\');
            return (std::filesystem::path(root) / relative).wstring();
        }

        // decodes a picture file into a bitmap at roughly the size it will be shown
        IAsyncAction LoadBitmapAsync(BitmapImage bitmap, std::wstring path, Microsoft::UI::Dispatching::DispatcherQueue dispatcher)
        {
            auto file = co_await StorageFile::GetFileFromPathAsync(path);
            auto stream = co_await file.OpenReadAsync();
            co_await wil::resume_foreground(dispatcher);
            if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile
            co_await bitmap.SetSourceAsync(stream);
        }
    }

    CreatePage::CreatePage()
    {
        InitializeComponent();
        // re-arrange the two panes whenever the page width crosses the stacking threshold
        PageScroll().SizeChanged([weak = get_weak()](auto&&, SizeChangedEventArgs const& args)
        {
            if (auto self = weak.get(); self && !aiwf::AppState::Get().ShuttingDown()) self->ApplyLayout(args.NewSize().Width);
        });
    }

    // ---- layout: side by side when wide, one scrolling column when narrow ----------------------------------------
    void CreatePage::ApplyLayout(double width)
    {
        bool stacked = width < StackBelowWidth;
        if (m_layoutKnown && stacked == m_stacked) return;
        m_layoutKnown = true;
        m_stacked = stacked;
        // the outer scroller only scrolls in the stacked layout; side by side, each pane fits the window
        PageScroll().VerticalScrollMode(stacked ? ScrollMode::Enabled : ScrollMode::Disabled);
        PageScroll().VerticalScrollBarVisibility(stacked ? ScrollBarVisibility::Auto : ScrollBarVisibility::Disabled);
        FormScroll().VerticalScrollMode(stacked ? ScrollMode::Disabled : ScrollMode::Enabled);
        FormColumn().Width(stacked ? GridLengthHelper::FromValueAndType(1, GridUnitType::Star) : GridLengthHelper::FromPixels(340));
        ResultColumn().Width(stacked ? GridLengthHelper::FromPixels(0) : GridLengthHelper::FromValueAndType(1, GridUnitType::Star));
        TopRow().Height(stacked ? GridLengthHelper::Auto() : GridLengthHelper::FromValueAndType(1, GridUnitType::Star));
        RootGrid().ColumnSpacing(stacked ? 0 : 28);
        RootGrid().RowSpacing(stacked ? 24 : 0);
        Grid::SetRow(ResultPane(), stacked ? 1 : 0);
        Grid::SetColumn(ResultPane(), stacked ? 0 : 1);
        // stacked, the preview needs a real height of its own instead of "the rest of the window"
        PreviewRow().Height(stacked ? GridLengthHelper::FromPixels(340) : GridLengthHelper::FromValueAndType(1, GridUnitType::Star));
    }

    // ---- page lifetime ------------------------------------------------------------------------------------
    void CreatePage::OnNavigatedTo(NavigationEventArgs const&)
    {
        m_recentLoaded = false;
        auto weak = get_weak();
        auto dispatcher = DispatcherQueue();
        m_statusToken = aiwf::EngineSupervisor::Instance().StatusChanged([weak, dispatcher]
        {
            dispatcher.TryEnqueue([weak]
            {
                if (auto self = weak.get(); self && !aiwf::AppState::Get().ShuttingDown()) self->UpdateEngineState();
            });
        });
        UpdateEngineState();
    }

    void CreatePage::OnNavigatedFrom(NavigationEventArgs const&)
    {
        aiwf::EngineSupervisor::Instance().StatusChanged(m_statusToken);
    }

    void CreatePage::UpdateEngineState()
    {
        bool ready = ui::UpdateEngineBar(EngineBar(), { L"engine-api", L"image" });
        GenerateButton().IsEnabled(ready && !m_busy);
        if (!m_recentLoaded && aiwf::EngineSupervisor::Instance().IsAnswering(L"engine-api")) LoadRecentAsync();
    }

    void CreatePage::StartEngine_Click(IInspectable const&, RoutedEventArgs const&)
    {
        ui::StartEnginesAsync({ L"engine-api", L"image" });
    }

    void CreatePage::SetBusy(bool busy)
    {
        m_busy = busy;
        GenerateButton().IsEnabled(!busy && aiwf::EngineSupervisor::Instance().IsAnswering(L"engine-api") &&
                                   aiwf::EngineSupervisor::Instance().IsAnswering(L"image"));
        CancelButton().Visibility(busy ? Visibility::Visible : Visibility::Collapsed);
        JobPanel().Visibility(busy || !JobStatus().Text().empty() ? Visibility::Visible : Visibility::Collapsed);
        JobProgress().Visibility(busy ? Visibility::Visible : Visibility::Collapsed);
    }

    // ---- generating -----------------------------------------------------------------------------------------
    void CreatePage::Generate_Click(IInspectable const&, RoutedEventArgs const&)
    {
        GenerateAsync();
    }

    fire_and_forget CreatePage::GenerateAsync()
    {
        if (m_busy) co_return;
        std::wstring prompt{ PromptBox().Text() };
        while (!prompt.empty() && iswspace(prompt.back())) prompt.pop_back();
        while (!prompt.empty() && iswspace(prompt.front())) prompt.erase(0, 1);
        if (prompt.empty())
        {
            ErrorBar().Title(L"Nothing to make yet");
            ErrorBar().Message(L"Describe the image first.");
            ErrorBar().IsOpen(true);
            co_return;
        }
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        auto& state = aiwf::AppState::Get();
        ErrorBar().IsOpen(false);

        // this block turns the form into the engine API's generate_image request
        auto aspect = ui::SelectedTag(AspectBox());
        JsonObject body;
        body.Insert(L"prompt", JsonValue::CreateStringValue(prompt));
        body.Insert(L"aspect_ratio", JsonValue::CreateStringValue(aspect.empty() ? hstring(L"1:1") : aspect));
        body.Insert(L"quality", JsonValue::CreateStringValue(QualityBox().SelectedIndex() == 1 ? L"standard" : L"draft"));
        double seedValue = SeedBox().Value();
        if (!std::isnan(seedValue) && seedValue >= 0) body.Insert(L"seed", JsonValue::CreateNumberValue(std::floor(seedValue)));
        if (!state.ProjectId().empty()) body.Insert(L"project_id", JsonValue::CreateStringValue(state.ProjectId()));

        m_disconnected = false;
        m_dismissRequested = false;
        DismissButton().Visibility(Visibility::Collapsed);
        DismissButton().IsEnabled(true);
        SetBusy(true);
        JobProgress().IsIndeterminate(true);
        JobStatus().Text(L"Sending to the image engine...");

        JsonObject job{ nullptr };
        std::wstring outputDir;
        hstring failure;
        try
        {
            auto root = co_await http::GetJsonAsync(state.EngineApiRoot() + L"/");
            outputDir = std::wstring(http::Str(root, L"output_dir"));
            if (outputDir.empty()) throw hresult_error(E_FAIL, L"The engine API returned no output folder.");
            auto started = co_await http::PostJsonAsync(state.EngineApi() + L"/images", body);
            job = http::Obj(started, L"job");
        }
        catch (hresult_error const& error)
        {
            failure = error.message();
        }
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile
        if (!failure.empty() || !job)
        {
            JobStatus().Text(L"The image was not started.");
            SetBusy(false);
            ErrorBar().Title(L"The image was not started");
            ErrorBar().Message(failure.empty() ? hstring(L"The engine API returned no job.") : failure);
            ErrorBar().IsOpen(true);
            co_return;
        }
        m_jobId = http::Str(job, L"job_id");
        auto jobUrl = state.EngineApi() + L"/images/" + http::EscapeSegment(hstring(m_jobId));

        // this loop follows the job once a second until it is done, failed or cancelled
        std::wstring finalState;
        for (;;)
        {
            co_await resume_after(std::chrono::seconds(1));
            failure = L"";
            try
            {
                job = http::Obj(co_await http::GetJsonAsync(jobUrl), L"job");
            }
            catch (hresult_error const& error)
            {
                failure = error.message();
            }
            co_await wil::resume_foreground(dispatcher);
            if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile
            if (m_dismissRequested)
            {
                finalState = L"dismissed";
                break;
            }
            if (!failure.empty())
            {
                m_disconnected = true;
                DismissButton().Visibility(Visibility::Visible);
                JobStatus().Text(L"Connection lost; retrying automatically. Cancel remains available. " + failure);
                continue;
            }
            if (!job)
            {
                m_disconnected = true;
                DismissButton().Visibility(Visibility::Visible);
                JobStatus().Text(L"The engine returned no job status; retrying automatically.");
                continue;
            }
            m_disconnected = false;
            DismissButton().Visibility(Visibility::Collapsed);
            auto stateName = std::wstring(http::Str(job, L"state"));
            auto elapsed = static_cast<int>(http::Num(job, L"elapsed_seconds"));
            if (stateName == L"queued")
            {
                auto position = static_cast<int>(http::Num(job, L"queue_position"));
                JobStatus().Text(position > 0 ? hstring(L"Waiting for the GPU (position " + std::to_wstring(position) + L" in the image queue)...")
                                              : hstring(L"Queued..."));
            }
            else if (stateName == L"running")
            {
                JobStatus().Text(hstring(L"Generating on the GPU... " + std::to_wstring(elapsed) +
                                         L" s  (the first image after a restart also loads about 20 GB of weights)"));
            }
            else
            {
                finalState = stateName;
                break;
            }
        }

        // this block shows the outcome
        SetBusy(false);
        DismissButton().Visibility(Visibility::Collapsed);
        if (finalState == L"done")
        {
            auto images = http::Arr(job, L"images");
            if (images.Size() > 0)
            {
                auto image = images.GetObjectAt(0);
                Shown shown;
                shown.path = JoinOutput(outputDir, std::wstring(http::Str(image, L"relative_path")));
                shown.prompt = prompt;
                shown.aspect = std::wstring(http::Str(job, L"aspect_ratio"));
                shown.seed = static_cast<int64_t>(http::Num(job, L"seed", -1));
                wchar_t info[160]{};
                swprintf_s(info, L"%d × %d  ·  seed %lld  ·  %.0f s  ·  %s", static_cast<int>(http::Num(image, L"width")),
                           static_cast<int>(http::Num(image, L"height")), shown.seed, http::Num(job, L"elapsed_seconds"),
                           http::Str(job, L"model").c_str());
                ShowImageAsync(shown, info);
                JobStatus().Text(L"Done.");
                LoadRecentAsync();
            }
        }
        else if (finalState == L"failed")
        {
            JobStatus().Text(L"The image failed.");
            auto error = http::Obj(job, L"error");
            ErrorBar().Title(L"The image failed");
            ErrorBar().Message(error ? http::Str(error, L"message") : hstring(L"See the engine log in Settings."));
            ErrorBar().IsOpen(true);
        }
        else if (finalState == L"dismissed") JobStatus().Text(L"Tracking dismissed. The image engine may still be working.");
        else if (finalState == L"cancelled") JobStatus().Text(L"Image cancelled.");
        m_jobId.clear();
    }

    void CreatePage::Dismiss_Click(IInspectable const&, RoutedEventArgs const&)
    {
        if (!m_busy || !m_disconnected || m_jobId.empty()) return;
        m_dismissRequested = true;
        DismissButton().IsEnabled(false);
        JobStatus().Text(L"Dismissing tracking. The image engine may still be working.");
    }

    fire_and_forget CreatePage::Cancel_Click(IInspectable const&, RoutedEventArgs const&)
    {
        if (m_jobId.empty()) co_return;
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        auto jobId = m_jobId;
        auto url = aiwf::AppState::Get().EngineApi() + L"/images/" + http::EscapeSegment(hstring(m_jobId)) + L"/cancel";
        JobStatus().Text(L"Cancelling...");
        hstring failure;
        try
        {
            co_await http::PostJsonAsync(url, JsonObject());
        }
        catch (hresult_error const& error)
        {
            failure = error.message();
        }
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown() || jobId != m_jobId) co_return;
        if (!failure.empty())
        {
            ErrorBar().Title(L"Cancellation was not confirmed");
            ErrorBar().Message(failure);
            ErrorBar().IsOpen(true);
            JobStatus().Text(L"Tracking continues. Retry Cancel when the engine is available.");
        }
    }

    // ---- showing images ----------------------------------------------------------------------------------------
    fire_and_forget CreatePage::ShowImageAsync(Shown shown, hstring info)
    {
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        BitmapImage bitmap;
        hstring failure;
        try
        {
            co_await LoadBitmapAsync(bitmap, shown.path, dispatcher);
        }
        catch (hresult_error const& error)
        {
            failure = error.message();
        }
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile
        if (!failure.empty())
        {
            PreviewInfo().Text(L"Could not open " + hstring(shown.path) + L": " + failure);
            co_return;
        }
        Preview().Source(bitmap);
        PreviewEmpty().Visibility(Visibility::Collapsed);
        PreviewInfo().Text(info);
        m_shown = std::move(shown);
        OpenButton().IsEnabled(true);
        FolderButton().IsEnabled(true);
        ReuseButton().IsEnabled(!m_shown.prompt.empty());
    }

    fire_and_forget CreatePage::LoadRecentAsync()
    {
        m_recentLoaded = true;
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        auto& state = aiwf::AppState::Get();
        JsonObject root{ nullptr };
        JsonObject listing{ nullptr };
        try
        {
            // the output folder comes from the engine API, so this page and Studio agree on it
            root = co_await http::GetJsonAsync(state.EngineApiRoot() + L"/");
            listing = co_await http::GetJsonAsync(state.EngineApi() + L"/outputs?limit=24");
        }
        catch (hresult_error const&)
        {
            m_recentLoaded = false;   // try again on the next engine status change
        }
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile
        if (root) m_outputDir = std::wstring(http::Str(root, L"output_dir"));
        if (!listing || m_outputDir.empty()) co_return;

        Recent().Children().Clear();
        auto weak = get_weak();
        // this loop adds one thumbnail button per recent output, newest first
        for (auto const& value : http::Arr(listing, L"outputs"))
        {
            auto output = value.GetObject();
            Shown shown;
            shown.path = JoinOutput(m_outputDir, std::wstring(http::Str(output, L"relative_path")));
            shown.prompt = std::wstring(http::Str(output, L"prompt"));
            shown.seed = static_cast<int64_t>(http::Num(output, L"seed", -1));
            auto width = http::Num(output, L"width", 1);
            auto height = http::Num(output, L"height", 1);

            Image thumb;
            thumb.Height(120);
            thumb.Width(height > 0 ? 120.0 * width / height : 120);
            thumb.Stretch(Media::Stretch::UniformToFill);
            BitmapImage bitmap;
            bitmap.DecodePixelHeight(240);
            thumb.Source(bitmap);
            [](BitmapImage target, std::wstring path, Microsoft::UI::Dispatching::DispatcherQueue queue) -> fire_and_forget
            {
                try
                {
                    co_await LoadBitmapAsync(target, path, queue);
                }
                catch (hresult_error const&)
                {
                    // a file that moved or vanished just shows an empty tile
                }
            }(bitmap, shown.path, dispatcher);

            Button tile;
            tile.Padding(ThicknessHelper::FromUniformLength(0));
            tile.CornerRadius(CornerRadiusHelper::FromUniformRadius(6));
            tile.Content(thumb);
            ToolTipService::SetToolTip(tile, box_value(shown.prompt.empty() ? hstring(shown.path) : hstring(shown.prompt)));
            Automation::AutomationProperties::SetName(tile, hstring(shown.prompt.empty() ? L"Recent image" : shown.prompt));
            wchar_t info[160]{};
            swprintf_s(info, L"%d × %d  ·  seed %lld", static_cast<int>(width), static_cast<int>(height), shown.seed);
            tile.Click([weak, shown, text = std::wstring(info)](auto&&, auto&&)
            {
                if (auto self = weak.get(); self && !aiwf::AppState::Get().ShuttingDown()) self->ShowImageAsync(shown, hstring(text));
            });
            Recent().Children().Append(tile);
        }
    }

    // ---- actions on the shown image ---------------------------------------------------------------------------
    void CreatePage::Open_Click(IInspectable const&, RoutedEventArgs const&)
    {
        if (!m_shown.path.empty()) ui::OpenWithShell(m_shown.path);
    }

    void CreatePage::ShowInFolder_Click(IInspectable const&, RoutedEventArgs const&)
    {
        if (!m_shown.path.empty()) ui::ShowInExplorer(m_shown.path);
    }

    void CreatePage::Reuse_Click(IInspectable const&, RoutedEventArgs const&)
    {
        PromptBox().Text(m_shown.prompt);
        if (m_shown.seed >= 0) SeedBox().Value(static_cast<double>(m_shown.seed));
        // re-select the shape when the image came from this page and the shape is known
        for (uint32_t i = 0; i < AspectBox().Items().Size(); ++i)
        {
            auto item = AspectBox().Items().GetAt(i).try_as<ComboBoxItem>();
            if (item && unbox_value_or<hstring>(item.Tag(), L"") == m_shown.aspect) AspectBox().SelectedIndex(static_cast<int32_t>(i));
        }
    }
}
