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
        // Studio's output folder + a relative path from the engine API
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
        m_creativeReady = true;
        PopulateBackendPicker();
        PopulateModelPicker();
        UpdateBackendStatus();
        UpdateCreateControls();
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
        ++m_backendGeneration;
        m_backendsLoading = false;
        m_backendsLoaded = false;
        m_backendLoadAttempted = false;
        m_backendLoadError.clear();
        m_preferenceSaveError.clear();
        m_backends.clear();
        PopulateBackendPicker();
        PopulateModelPicker();
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
        ++m_backendGeneration;
        m_backendsLoading = false;
        m_backendLoadAttempted = false;
    }

    void CreatePage::UpdateEngineState()
    {
        bool ready = ui::UpdateEngineBar(EngineBar(), { L"engine-api" });
        if (!ready)
        {
            InvalidateBackendDiscovery();
        }
        else
        {
            if (!m_recentLoaded) LoadRecentAsync();
            if (!m_backendLoadAttempted && !m_backendsLoading) LoadBackendsAsync(m_backendGeneration);
        }
        UpdateCreateControls();
    }

    void CreatePage::InvalidateBackendDiscovery()
    {
        ++m_backendGeneration;
        m_backendsLoading = false;
        m_backendsLoaded = false;
        m_backendLoadAttempted = false;
        m_backendLoadError.clear();
        m_backends.clear();
        PopulateBackendPicker();
        PopulateModelPicker();
    }

    void CreatePage::StartEngine_Click(IInspectable const&, RoutedEventArgs const&)
    {
        StartSelectedBackend();
    }

    void CreatePage::StartSelectedBackend()
    {
        // Both image backends run through AIWF's own engine API and Studio environment.
        ui::StartEnginesAsync({ L"engine-api" });
    }

    void CreatePage::SetBusy(bool busy)
    {
        m_busy = busy;
        UpdateCreateControls();
        CancelButton().Visibility(busy ? Visibility::Visible : Visibility::Collapsed);
        JobPanel().Visibility(busy || !JobStatus().Text().empty() ? Visibility::Visible : Visibility::Collapsed);
        JobProgress().Visibility(busy ? Visibility::Visible : Visibility::Collapsed);
    }

    CreatePage::ImageBackendOption const* CreatePage::FindBackend(std::wstring const& id) const
    {
        auto found = std::find_if(m_backends.begin(), m_backends.end(), [&](auto const& backend)
        {
            return backend.id == id;
        });
        return found == m_backends.end() ? nullptr : &*found;
    }

    CreatePage::ImageModelOption const* CreatePage::FindModel(ImageBackendOption const& backend, std::wstring const& id) const
    {
        auto found = std::find_if(backend.models.begin(), backend.models.end(), [&](auto const& model)
        {
            return model.id == id && std::find(model.capabilities.begin(), model.capabilities.end(), L"text_to_image") != model.capabilities.end();
        });
        return found == backend.models.end() ? nullptr : &*found;
    }

    void CreatePage::PopulateBackendPicker()
    {
        auto selectedId = aiwf::AppState::Get().ImageBackendId();
        m_populatingSelectors = true;
        ImageBackendPicker().Items().Clear();
        std::set<std::wstring> added;
        auto add = [&](std::wstring const& id, std::wstring const& label, std::wstring const& reason)
        {
            ComboBoxItem item;
            item.Content(box_value(hstring(label)));
            item.Tag(box_value(hstring(id)));
            if (!reason.empty()) ToolTipService::SetToolTip(item, box_value(hstring(reason)));
            ImageBackendPicker().Items().Append(item);
            added.insert(id);
        };

        auto enabled = aiwf::AppState::Get().EnabledImageBackends();
        if (m_backendsLoaded)
        {
            for (auto const& id : enabled)
            {
                auto backend = FindBackend(id);
                if (backend)
                {
                    auto label = backend->label.empty() ? backend->id : backend->label;
                    add(backend->id, label, backend->reason);
                }
                else
                {
                    auto label = id == L"python" ? L"AIWF Python (Diffusers)" : L"AIWF C++ (stable-diffusion.cpp)";
                    add(id, label, L"The running engine did not report this selected backend. Update the engine service and refresh choices.");
                }
            }
        }
        else
        {
            for (auto const& id : enabled)
            {
                auto label = id == L"python" ? L"AIWF Python (Diffusers)" : L"AIWF C++ (stable-diffusion.cpp)";
                add(id, label, L"");
            }
        }

        if (std::find(enabled.begin(), enabled.end(), selectedId) != enabled.end() && !added.contains(selectedId))
        {
            add(selectedId, L"Saved backend (" + selectedId + L")", L"The running engine does not report this saved backend. Choose a supported backend.");
        }
        ImageBackendPicker().SelectedIndex(-1);
        for (uint32_t i = 0; i < ImageBackendPicker().Items().Size(); ++i)
        {
            auto item = ImageBackendPicker().Items().GetAt(i).try_as<ComboBoxItem>();
            if (item && unbox_value_or<hstring>(item.Tag(), L"") == hstring(selectedId))
            {
                ImageBackendPicker().SelectedIndex(static_cast<int32_t>(i));
                break;
            }
        }
        m_populatingSelectors = false;
    }

    void CreatePage::PopulateModelPicker()
    {
        auto backendId = std::wstring(ui::SelectedTag(ImageBackendPicker()));
        auto savedModelId = backendId.empty() ? std::wstring{} : aiwf::AppState::Get().ImageModelId(backendId);
        int32_t selectedIndex = -1;
        m_populatingSelectors = true;
        ImageModelPicker().Items().Clear();
        ImageModelPicker().SelectedIndex(-1);
        ImageModelPicker().PlaceholderText(L"Choose an available model");

        auto backend = FindBackend(backendId);
        if (m_backendsLoaded && backend)
        {
            for (auto const& model : backend->models)
            {
                if (std::find(model.capabilities.begin(), model.capabilities.end(), L"text_to_image") == model.capabilities.end()) continue;
                ComboBoxItem item;
                item.Content(box_value(hstring(model.label.empty() ? model.id : model.label)));
                item.Tag(box_value(hstring(model.id)));
                std::wstring tooltip = model.family;
                for (auto const& capability : model.capabilities)
                {
                    if (!tooltip.empty()) tooltip += L" · ";
                    tooltip += capability;
                }
                if (!tooltip.empty()) ToolTipService::SetToolTip(item, box_value(hstring(tooltip)));
                ImageModelPicker().Items().Append(item);
                if (!savedModelId.empty() && model.id == savedModelId)
                {
                    selectedIndex = static_cast<int32_t>(ImageModelPicker().Items().Size() - 1);
                }
            }
        }
        if (selectedIndex >= 0) ImageModelPicker().SelectedIndex(selectedIndex);
        m_populatingSelectors = false;

        UpdateBackendStatus();
        UpdateCreateControls();
    }

    void CreatePage::UpdateBackendStatus()
    {
        if (!m_preferenceSaveError.empty())
        {
            BackendStatusText().Text(hstring(m_preferenceSaveError));
            return;
        }
        if (!aiwf::EngineSupervisor::Instance().IsAnswering(L"engine-api"))
        {
            BackendStatusText().Text(L"Start the engine API to check backend availability and load models.");
            return;
        }
        if (m_backendsLoading)
        {
            BackendStatusText().Text(L"Checking backend availability and compatible models...");
            return;
        }
        if (!m_backendsLoaded)
        {
            BackendStatusText().Text(m_backendLoadError.empty()
                ? L"Refresh choices to find available backends and model files."
                : hstring(L"Could not load backend choices: " + m_backendLoadError));
            return;
        }

        auto backendId = std::wstring(ui::SelectedTag(ImageBackendPicker()));
        auto backend = FindBackend(backendId);
        if (!backend)
        {
            BackendStatusText().Text(L"The selected backend is not reported by the running engine. Choose a supported backend or update the engine service.");
            return;
        }
        if (!backend->available)
        {
            BackendStatusText().Text(backend->reason.empty()
                ? L"This backend is unavailable. Check its setup and try Refresh choices."
                : hstring(backend->reason));
            return;
        }
        auto modelId = std::wstring(ui::SelectedTag(ImageModelPicker()));
        if (!modelId.empty())
        {
            if (auto model = FindModel(*backend, modelId))
            {
                BackendStatusText().Text(hstring(L"Selected: " + backend->label + L" · " + model->label +
                    L". This screen does not test model loading or generation."));
                return;
            }
            BackendStatusText().Text(L"The saved model is not available for this backend. Choose a listed model.");
            return;
        }
        if (!aiwf::AppState::Get().ImageModelId(backendId).empty())
        {
            BackendStatusText().Text(L"The saved model is not available for this backend. Choose a listed model.");
            return;
        }
        if (ImageModelPicker().Items().Size() == 0)
        {
            BackendStatusText().Text(L"No compatible text-to-image models are available for this backend.");
            return;
        }
        BackendStatusText().Text(L"Choose an available text-to-image model.");
    }

    void CreatePage::UpdateCreateControls()
    {
        auto backendId = std::wstring(ui::SelectedTag(ImageBackendPicker()));
        auto modelId = std::wstring(ui::SelectedTag(ImageModelPicker()));
        auto backend = m_backendsLoaded ? FindBackend(backendId) : nullptr;
        auto model = backend && !modelId.empty() ? FindModel(*backend, modelId) : nullptr;
        bool apiReady = aiwf::EngineSupervisor::Instance().IsAnswering(L"engine-api");
        bool hasModels = backend && std::any_of(backend->models.begin(), backend->models.end(), [](auto const& candidate)
        {
            return std::find(candidate.capabilities.begin(), candidate.capabilities.end(), L"text_to_image") != candidate.capabilities.end();
        });
        auto enabledBackends = aiwf::AppState::Get().EnabledImageBackends();
        ImageBackendPicker().IsEnabled(!m_busy && enabledBackends.size() > 1);
        ImageModelPicker().IsEnabled(!m_busy && m_backendsLoaded && hasModels);
        RefreshBackendsButton().IsEnabled(!m_busy && apiReady && !m_backendsLoading);
        GenerateButton().IsEnabled(!m_busy && apiReady && m_backendsLoaded && backend && backend->available && model);
        UpdateBackendStatus();
    }

    void CreatePage::ImageBackendPicker_SelectionChanged(IInspectable const&, SelectionChangedEventArgs const&)
    {
        if (m_populatingSelectors) return;
        auto selectedId = std::wstring(ui::SelectedTag(ImageBackendPicker()));
        if (selectedId.empty()) return;
        auto& state = aiwf::AppState::Get();
        if (!state.SetImageBackendId(selectedId))
        {
            m_preferenceSaveError = L"The backend choice could not be saved. Check %LOCALAPPDATA%\\Programs\\AES Studio Data\\settings.json; the previous choice remains selected.";
            PopulateBackendPicker();
            PopulateModelPicker();
            return;
        }
        m_preferenceSaveError.clear();
        PopulateModelPicker();
    }

    void CreatePage::ImageModelPicker_SelectionChanged(IInspectable const&, SelectionChangedEventArgs const&)
    {
        if (m_populatingSelectors) return;
        auto backendId = std::wstring(ui::SelectedTag(ImageBackendPicker()));
        auto modelId = std::wstring(ui::SelectedTag(ImageModelPicker()));
        if (backendId.empty() || modelId.empty()) return;
        auto& state = aiwf::AppState::Get();
        if (!state.SetImageModelId(backendId, modelId))
        {
            auto previous = state.ImageModelId(backendId);
            m_preferenceSaveError = L"The model choice could not be saved. Check %LOCALAPPDATA%\\Programs\\AES Studio Data\\settings.json; the previous choice remains selected.";
            m_populatingSelectors = true;
            ImageModelPicker().SelectedIndex(-1);
            for (uint32_t i = 0; i < ImageModelPicker().Items().Size(); ++i)
            {
                auto item = ImageModelPicker().Items().GetAt(i).try_as<ComboBoxItem>();
                if (item && unbox_value_or<hstring>(item.Tag(), L"") == hstring(previous))
                {
                    ImageModelPicker().SelectedIndex(static_cast<int32_t>(i));
                    break;
                }
            }
            m_populatingSelectors = false;
        }
        else
        {
            m_preferenceSaveError.clear();
        }
        UpdateCreateControls();
    }

    void CreatePage::RefreshBackends_Click(IInspectable const&, RoutedEventArgs const&)
    {
        if (!aiwf::EngineSupervisor::Instance().IsAnswering(L"engine-api"))
        {
            InvalidateBackendDiscovery();
            UpdateCreateControls();
            return;
        }
        m_backendLoadAttempted = false;
        m_backendLoadError.clear();
        LoadBackendsAsync(m_backendGeneration);
    }

    fire_and_forget CreatePage::LoadBackendsAsync(uint64_t generation)
    {
        if (m_backendsLoading || !aiwf::EngineSupervisor::Instance().IsAnswering(L"engine-api")) co_return;
        m_backendLoadAttempted = true;
        m_backendsLoading = true;
        m_backendsLoaded = false;
        m_backendLoadError.clear();
        m_backends.clear();
        PopulateBackendPicker();
        PopulateModelPicker();

        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        auto& state = aiwf::AppState::Get();
        auto enabledBackends = state.EnabledImageBackends();
        std::vector<ImageBackendOption> discovered;
        hstring failure;
        try
        {
            std::wstring url = std::wstring(state.EngineApi()) + L"/images/backends";
            std::wstring separator = L"?";
            for (auto const& id : enabledBackends)
            {
                url += separator + L"enabled_backend_ids=" + id;
                separator = L"&";
            }
            auto response = co_await http::GetJsonAsync(hstring(url));
            if (http::Str(response, L"schema_version") != L"1")
            {
                throw hresult_error(E_FAIL, L"The image backend list uses an unsupported schema. Update the engine service.");
            }
            auto backends = http::Arr(response, L"backends");
            for (auto const& value : backends)
            {
                if (value.ValueType() != JsonValueType::Object) continue;
                auto item = value.GetObject();
                ImageBackendOption backend;
                backend.id = http::Str(item, L"id");
                backend.label = http::Str(item, L"label");
                backend.reason = http::Str(item, L"reason");
                backend.available = http::Bool(item, L"available");
                if (backend.id.empty() || backend.label.empty()) continue;
                if (std::find(enabledBackends.begin(), enabledBackends.end(), backend.id) == enabledBackends.end()) continue;
                for (auto const& modelValue : http::Arr(item, L"models"))
                {
                    if (modelValue.ValueType() != JsonValueType::Object) continue;
                    auto modelObject = modelValue.GetObject();
                    ImageModelOption model;
                    model.id = http::Str(modelObject, L"id");
                    model.label = http::Str(modelObject, L"label");
                    model.family = http::Str(modelObject, L"family");
                    for (auto const& capabilityValue : http::Arr(modelObject, L"capabilities"))
                    {
                        if (capabilityValue.ValueType() == JsonValueType::String)
                        {
                            model.capabilities.push_back(std::wstring(capabilityValue.GetString()));
                        }
                    }
                    if (!model.id.empty() && !model.label.empty()) backend.models.push_back(std::move(model));
                }
                discovered.push_back(std::move(backend));
            }
            if (discovered.empty()) throw hresult_error(E_FAIL, L"The engine returned no image backends. Update the engine service and refresh choices.");
        }
        catch (hresult_error const& error)
        {
            failure = error.message();
        }
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown() || generation != m_backendGeneration) co_return;
        m_backendsLoading = false;
        if (!failure.empty())
        {
            m_backendLoadError = std::wstring(failure);
            PopulateBackendPicker();
            PopulateModelPicker();
            co_return;
        }
        m_backends = std::move(discovered);
        m_backendsLoaded = true;
        m_backendLoadError.clear();
        PopulateBackendPicker();
        PopulateModelPicker();
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
        auto backendId = std::wstring(ui::SelectedTag(ImageBackendPicker()));
        auto modelId = std::wstring(ui::SelectedTag(ImageModelPicker()));
        auto backend = m_backendsLoaded ? FindBackend(backendId) : nullptr;
        auto selectedModel = backend && !modelId.empty() ? FindModel(*backend, modelId) : nullptr;
        if (!backend || !backend->available || !selectedModel)
        {
            UpdateBackendStatus();
            ErrorBar().Title(L"Choose a ready image backend and model");
            ErrorBar().Message(BackendStatusText().Text());
            ErrorBar().IsOpen(true);
            co_return;
        }
        auto backendLabel = backend->label;
        auto modelLabel = selectedModel->label;
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        auto& state = aiwf::AppState::Get();
        ErrorBar().IsOpen(false);
        ErrorBar().Severity(InfoBarSeverity::Error);

        // this block turns the form into the engine API's generate_image request
        auto aspect = ui::SelectedTag(AspectBox());
        JsonObject body;
        body.Insert(L"prompt", JsonValue::CreateStringValue(prompt));
        body.Insert(L"backend_id", JsonValue::CreateStringValue(backendId));
        body.Insert(L"model_id", JsonValue::CreateStringValue(modelId));
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
            auto setup = co_await http::GetJsonAsync(state.EngineApi() + L"/setup");
            auto output = http::Obj(http::Obj(setup, L"folders"), L"output_dir");
            outputDir = std::wstring(http::Str(output, L"path"));
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
                JobStatus().Text(hstring(L"Generating image... " + std::to_wstring(elapsed) + L" s"));
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
            std::wstring warningText;
            for (auto const& value : http::Arr(job, L"warnings"))
            {
                auto message = std::wstring(http::Str(value.GetObject(), L"message"));
                if (message.empty()) continue;
                if (!warningText.empty()) warningText += L" ";
                warningText += message;
            }
            if (images.Size() > 0)
            {
                auto image = images.GetObjectAt(0);
                Shown shown;
                shown.path = JoinOutput(outputDir, std::wstring(http::Str(image, L"relative_path")));
                shown.prompt = prompt;
                shown.aspect = std::wstring(http::Str(job, L"aspect_ratio"));
                shown.backendId = backendId;
                shown.modelId = modelId;
                shown.seed = static_cast<int64_t>(http::Num(job, L"seed", -1));
                auto info = std::to_wstring(static_cast<int>(http::Num(image, L"width"))) + L" × " +
                            std::to_wstring(static_cast<int>(http::Num(image, L"height"))) + L"  ·  seed " +
                            std::to_wstring(shown.seed) + L"  ·  " + backendLabel + L" · " + modelLabel;
                ShowImageAsync(shown, hstring(info));
                JobStatus().Text(warningText.empty() ? hstring(L"Done.") : hstring(L"Done. " + warningText));
                if (!warningText.empty())
                {
                    ErrorBar().Severity(InfoBarSeverity::Warning);
                    ErrorBar().Title(L"Image saved; project history may be incomplete");
                    ErrorBar().Message(hstring(warningText));
                    ErrorBar().IsOpen(true);
                }
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
        JsonObject setup{ nullptr };
        JsonObject listing{ nullptr };
        try
        {
            setup = co_await http::GetJsonAsync(state.EngineApi() + L"/setup");
            listing = co_await http::GetJsonAsync(state.EngineApi() + L"/outputs?limit=24");
        }
        catch (hresult_error const&)
        {
            m_recentLoaded = false;   // try again on the next engine status change
        }
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile
        if (setup) m_outputDir = std::wstring(http::Str(http::Obj(http::Obj(setup, L"folders"), L"output_dir"), L"path"));
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
            shown.backendId = std::wstring(http::Str(output, L"backend_id"));
            shown.modelId = std::wstring(http::Str(output, L"model_id"));
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
        if (m_shown.backendId.empty() || m_shown.modelId.empty())
        {
            ErrorBar().Title(L"Backend and model were not recorded");
            ErrorBar().Message(L"This image predates backend metadata, so its exact backend and model cannot be restored.");
            ErrorBar().IsOpen(true);
            return;
        }

        if (!aiwf::AppState::Get().IsImageBackendEnabled(m_shown.backendId))
        {
            ErrorBar().Title(L"The image backend is not enabled");
            ErrorBar().Message(L"Run setup again to enable this backend before restoring its settings.");
            ErrorBar().IsOpen(true);
            return;
        }

        auto backend = m_backendsLoaded ? FindBackend(m_shown.backendId) : nullptr;
        if (!backend)
        {
            ErrorBar().Title(L"The image backend is not reported");
            ErrorBar().Message(L"Refresh choices or update the engine service before restoring this image's settings.");
            ErrorBar().IsOpen(true);
            return;
        }
        if (!FindModel(*backend, m_shown.modelId))
        {
            ErrorBar().Title(L"The image model is not available");
            ErrorBar().Message(L"The model used for this image is not available for the selected backend. Refresh choices or select another model.");
            ErrorBar().IsOpen(true);
            return;
        }

        auto& state = aiwf::AppState::Get();
        if (!state.SetImageSelection(m_shown.backendId, m_shown.modelId))
        {
            m_preferenceSaveError = L"The backend and model choices could not be saved. Check %LOCALAPPDATA%\\Programs\\AES Studio Data\\settings.json; the previous choices remain selected.";
            UpdateCreateControls();
            return;
        }
        m_preferenceSaveError.clear();
        PopulateBackendPicker();
        PopulateModelPicker();
        PromptBox().Text(m_shown.prompt);
        if (m_shown.seed >= 0) SeedBox().Value(static_cast<double>(m_shown.seed));
        // re-select the shape when the image came from this page and the shape is known
        for (uint32_t i = 0; i < AspectBox().Items().Size(); ++i)
        {
            auto item = AspectBox().Items().GetAt(i).try_as<ComboBoxItem>();
            if (item && unbox_value_or<hstring>(item.Tag(), L"") == m_shown.aspect) AspectBox().SelectedIndex(static_cast<int32_t>(i));
        }
    }
    // Local ComfyUI creative workflows use the same authenticated AES engine API as Create.
    void CreatePage::CreativePreset_SelectionChanged(IInspectable const&, SelectionChangedEventArgs const&)
    {
        if (!m_creativeReady || m_creativeBusy) return;
        auto preset = std::wstring(ui::SelectedTag(CreativePreset()));
        CreativeAudioVae().Visibility(preset == L"ltx25-i2v" ? Visibility::Visible : Visibility::Collapsed);
        CreativeInputPanel().Visibility(preset == L"moss-image" ? Visibility::Collapsed : Visibility::Visible);
        CreativeModel().Items().Clear();
        CreativeEncoder().Items().Clear();
        CreativeVae().Items().Clear();
        CreativeAudioVae().Items().Clear();
        m_creativeInventoryPreset.clear();
        m_creativeInventoryPort = 0;
        CreativeAvailability().Text(L"Find local model files for this workflow.");
    }

    void CreatePage::CreativeSetBusy(bool busy)
    {
        m_creativeBusy = busy;
        CreativePreset().IsEnabled(!busy);
        CreativePort().IsEnabled(!busy);
        CreativeRefresh().IsEnabled(!busy);
        CreativeModel().IsEnabled(!busy);
        CreativeEncoder().IsEnabled(!busy);
        CreativeVae().IsEnabled(!busy);
        CreativeAudioVae().IsEnabled(!busy);
        CreativePrompt().IsEnabled(!busy);
        CreativeChooseImage().IsEnabled(!busy);
        CreativeRun().IsEnabled(!busy);
        CreativeCancel().Visibility(busy ? Visibility::Visible : Visibility::Collapsed);
        CreativeProgress().Visibility(busy ? Visibility::Visible : Visibility::Collapsed);
    }

    void CreatePage::CreativeRefresh_Click(IInspectable const&, RoutedEventArgs const&)
    {
        if (!m_creativeBusy) RefreshCreativeAsync();
    }

    fire_and_forget CreatePage::RefreshCreativeAsync()
    {
        auto preset = std::wstring(ui::SelectedTag(CreativePreset()));
        auto portValue = CreativePort().Value();
        if (preset.empty() || !std::isfinite(portValue) || portValue < 1 || portValue > 65535 || std::floor(portValue) != portValue)
        {
            CreativeAvailability().Text(L"Choose a workflow and enter a local port from 1 to 65535.");
            co_return;
        }
        int port = static_cast<int>(portValue);
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        CreativeRefresh().IsEnabled(false);
        CreativeRun().IsEnabled(false);
        CreativePreset().IsEnabled(false);
        CreativePort().IsEnabled(false);
        CreativeAvailability().Text(L"Checking local ComfyUI...");
        JsonObject inventory{ nullptr };
        hstring failure;
        try
        {
            auto url = aiwf::AppState::Get().EngineApi() + L"/creative/inventory?preset=" + http::EscapeSegment(hstring(preset)) + L"&port=" + to_hstring(port);
            inventory = co_await http::GetJsonAsync(url, {}, 30000);
        }
        catch (hresult_error const& error) { failure = error.message(); }
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown()) co_return;
        CreativeRefresh().IsEnabled(true);
        CreativeRun().IsEnabled(true);
        CreativePreset().IsEnabled(true);
        CreativePort().IsEnabled(true);
        m_creativeInventoryPreset.clear();
        m_creativeInventoryPort = 0;
        if (!failure.empty() || !inventory)
        {
            CreativeAvailability().Text(failure.empty() ? hstring(L"Could not inspect local ComfyUI.") : failure);
            co_return;
        }
        auto fields = http::Obj(inventory, L"fields");
        auto suggested = http::Obj(inventory, L"suggested");
        auto fill = [&](ComboBox const& box, wchar_t const* key)
        {
            box.Items().Clear();
            auto values = http::Arr(fields, key);
            auto preferred = http::Str(suggested, key);
            int32_t selected = -1;
            for (uint32_t i = 0; values && i < values.Size(); ++i)
            {
                if (values.GetAt(i).ValueType() != JsonValueType::String) continue;
                auto value = values.GetStringAt(i);
                ComboBoxItem item;
                item.Content(box_value(value));
                item.Tag(box_value(value));
                box.Items().Append(item);
                if (!preferred.empty() && value == preferred) selected = static_cast<int32_t>(box.Items().Size() - 1);
            }
            box.SelectedIndex(selected);
        };
        fill(CreativeModel(), L"model");
        fill(CreativeEncoder(), L"encoder");
        fill(CreativeVae(), L"vae");
        fill(CreativeAudioVae(), L"audio_vae");
        auto missing = http::Arr(inventory, L"missing_nodes");
        std::wstring message;
        for (uint32_t i = 0; missing && i < missing.Size(); ++i)
        {
            if (!message.empty()) message += L", ";
            message += std::wstring(missing.GetStringAt(i));
        }
        if (!message.empty()) CreativeAvailability().Text(hstring(L"Missing ComfyUI nodes: " + message));
        else CreativeAvailability().Text(L"Choose matching model components below. No models or nodes were downloaded.");
        m_creativeInventoryPreset = preset;
        m_creativeInventoryPort = port;
    }

    fire_and_forget CreatePage::CreativeChooseImage_Click(IInspectable const&, RoutedEventArgs const&)
    {
        auto strong = get_strong();
        winrt::Windows::Storage::Pickers::FileOpenPicker picker;
        picker.SuggestedStartLocation(winrt::Windows::Storage::Pickers::PickerLocationId::PicturesLibrary);
        for (auto ext : { L".png", L".jpg", L".jpeg", L".webp" }) picker.FileTypeFilter().Append(ext);
        HWND owner = nullptr;
        if (XamlRoot() && XamlRoot().ContentIslandEnvironment())
            owner = winrt::Microsoft::UI::GetWindowFromWindowId(XamlRoot().ContentIslandEnvironment().AppWindowId());
        if (!owner) owner = GetActiveWindow();
        try
        {
            picker.as<::IInitializeWithWindow>()->Initialize(owner);
            auto file = co_await picker.PickSingleFileAsync();
            if (aiwf::AppState::Get().ShuttingDown()) co_return;
            if (file)
            {
                m_creativeImage = std::wstring(file.Path());
                CreativeImagePath().Text(file.Path());
            }
        }
        catch (hresult_error const& error) { CreativeStatus().Text(error.message()); }
    }

    void CreatePage::CreativeRun_Click(IInspectable const&, RoutedEventArgs const&)
    {
        if (!m_creativeBusy) RunCreativeAsync();
    }

    fire_and_forget CreatePage::RunCreativeAsync()
    {
        auto preset = std::wstring(ui::SelectedTag(CreativePreset()));
        double value = CreativePort().Value();
        if (!std::isfinite(value) || value < 1 || value > 65535 || std::floor(value) != value ||
            m_creativeInventoryPreset != preset || m_creativeInventoryPort != static_cast<int>(value))
        {
            CreativeStatus().Text(L"Find local model files for this workflow and port first.");
            co_return;
        }
        auto model = ui::SelectedTag(CreativeModel());
        auto encoder = ui::SelectedTag(CreativeEncoder());
        auto vae = ui::SelectedTag(CreativeVae());
        auto audioVae = ui::SelectedTag(CreativeAudioVae());
        if (model.empty() || encoder.empty() || vae.empty() || (preset == L"ltx25-i2v" && audioVae.empty()))
        {
            CreativeStatus().Text(L"Choose each compatible model component first.");
            co_return;
        }
        if (CreativePrompt().Text().empty() || (preset != L"moss-image" && m_creativeImage.empty()))
        {
            CreativeStatus().Text(L"Enter a prompt and choose a reference image when required.");
            co_return;
        }
        int port = static_cast<int>(value);
        JsonObject selections;
        selections.Insert(L"model", JsonValue::CreateStringValue(model));
        selections.Insert(L"encoder", JsonValue::CreateStringValue(encoder));
        selections.Insert(L"vae", JsonValue::CreateStringValue(vae));
        if (preset == L"ltx25-i2v") selections.Insert(L"audio_vae", JsonValue::CreateStringValue(audioVae));
        JsonObject body;
        body.Insert(L"preset", JsonValue::CreateStringValue(hstring(preset)));
        body.Insert(L"prompt", JsonValue::CreateStringValue(CreativePrompt().Text()));
        body.Insert(L"port", JsonValue::CreateNumberValue(port));
        body.Insert(L"selections", selections.as<IJsonValue>());
        if (preset != L"moss-image") body.Insert(L"input_path", JsonValue::CreateStringValue(hstring(m_creativeImage)));
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        auto& state = aiwf::AppState::Get();
        CreativeSetBusy(true);
        CreativeCancel().IsEnabled(false);
        CreativeOpen().IsEnabled(false);
        CreativeStatus().Text(L"Sending the workflow to local ComfyUI...");
        JsonObject job{ nullptr };
        std::wstring outputDir;
        hstring failure;
        try
        {
            auto setup = co_await http::GetJsonAsync(state.EngineApi() + L"/setup");
            outputDir = std::wstring(http::Str(http::Obj(http::Obj(setup, L"folders"), L"output_dir"), L"path"));
            if (outputDir.empty()) throw hresult_error(E_FAIL, L"The AES Studio output folder is unavailable.");
            auto reply = co_await http::PostJsonAsync(state.EngineApi() + L"/creative/jobs", body, {}, 60000);
            job = http::Obj(reply, L"job");
        }
        catch (hresult_error const& error) { failure = error.message(); }
        co_await wil::resume_foreground(dispatcher);
        if (state.ShuttingDown()) co_return;
        if (!failure.empty() || !job)
        {
            CreativeSetBusy(false);
            CreativeStatus().Text(failure.empty() ? hstring(L"The workflow did not start.") : failure);
            co_return;
        }
        m_creativeJobId = std::wstring(http::Str(job, L"job_id"));
        m_creativeJobPort = port;
        CreativeCancel().IsEnabled(true);
        auto url = state.EngineApi() + L"/creative/jobs/" + http::EscapeSegment(hstring(m_creativeJobId)) + L"?port=" + to_hstring(port);
        for (;;)
        {
            co_await resume_after(std::chrono::seconds(1));
            failure = L"";
            int statusCode = 0;
            try { job = http::Obj(co_await http::GetJsonAsync(url), L"job"); }
            catch (hresult_error const& error) { failure = error.message(); statusCode = http::StatusOf(error); }
            co_await wil::resume_foreground(dispatcher);
            if (state.ShuttingDown()) co_return;
            if (!failure.empty())
            {
                if (statusCode == 404)
                {
                    CreativeStatus().Text(L"Tracking was lost; check ComfyUI because the job may still be running.");
                    break;
                }
                CreativeStatus().Text(L"Connection lost; retrying. The ComfyUI job may still be running.");
                continue;
            }
            if (!job) { CreativeStatus().Text(L"Waiting for job status..."); continue; }
            auto phase = std::wstring(http::Str(job, L"state"));
            if (phase == L"queued" || phase == L"running")
            {
                CreativeStatus().Text(hstring((phase == L"queued" ? L"Queued... " : L"Running... ") +
                    std::to_wstring(static_cast<int>(http::Num(job, L"elapsed_seconds"))) + L" s"));
                continue;
            }
            if (phase == L"done")
            {
                auto files = http::Arr(job, L"files");
                if (files && files.Size() > 0)
                {
                    auto relative = std::wstring(http::Str(files.GetObjectAt(0), L"relative_path"));
                    m_creativeResult = JoinOutput(outputDir, relative);
                    CreativeOpen().IsEnabled(!relative.empty());
                    CreativeStatus().Text(hstring(L"Saved: " + m_creativeResult));
                }
                else CreativeStatus().Text(L"The workflow finished without a result.");
            }
            else if (phase == L"cancelled") CreativeStatus().Text(L"The workflow was cancelled.");
            else
            {
                auto error = http::Obj(job, L"error");
                CreativeStatus().Text(error ? http::Str(error, L"message", L"The workflow failed.") : hstring(L"The workflow failed."));
            }
            break;
        }
        m_creativeJobId.clear();
        CreativeSetBusy(false);
    }

    fire_and_forget CreatePage::CreativeCancel_Click(IInspectable const&, RoutedEventArgs const&)
    {
        if (m_creativeJobId.empty()) co_return;
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        auto url = aiwf::AppState::Get().EngineApi() + L"/creative/jobs/" + http::EscapeSegment(hstring(m_creativeJobId)) +
                   L"/cancel?port=" + to_hstring(m_creativeJobPort);
        hstring failure;
        try { co_await http::PostJsonAsync(url, JsonObject{}); }
        catch (hresult_error const& error) { failure = error.message(); }
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown()) co_return;
        CreativeStatus().Text(failure.empty() ? hstring(L"Queued job cancellation requested.") : failure);
    }

    void CreatePage::CreativeOpen_Click(IInspectable const&, RoutedEventArgs const&)
    {
        if (!m_creativeResult.empty()) ui::OpenWithShell(m_creativeResult);
    }

}

