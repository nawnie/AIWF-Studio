// Train page implementation.
#include "pch.h"
#include "TrainPage.xaml.h"
#if __has_include("TrainPage.g.cpp")
#include "TrainPage.g.cpp"
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
namespace http = aiwf::http;

namespace winrt::AiwfDesktop::implementation
{
    namespace
    {
        // this is the chip color section for model and gate states
        constexpr uint32_t Good = 0xFF2DA44E;
        constexpr uint32_t Caution = 0xFFE3A21A;
        constexpr uint32_t Bad = 0xFFD13438;
        constexpr uint32_t Neutral = 0xFF8A8A8A;

        std::wstring SizeText(double sizeB)
        {
            wchar_t text[48]{};
            swprintf_s(text, sizeB < 1 ? L"%.0f M parameters" : L"%.1f B parameters", sizeB < 1 ? sizeB * 1000 : sizeB);
            return text;
        }

        // one gate or dependency line: explicit state, name, and the engine's detail
        UIElement CheckRow(std::wstring const& state, hstring const& title, hstring const& detail)
        {
            wchar_t const* glyph = L"";   // check
            wchar_t const* stateLabel = L"Ready";
            if (state == L"warning" || state == L"caution")
            {
                glyph = L"";   // warning
                stateLabel = L"Warning";
            }
            else if (state != L"ready" && state != L"ok" && state != L"available")
            {
                glyph = L"";   // cross
                stateLabel = state == L"missing" ? L"Missing" : L"Blocked";
            }
            Grid row;
            row.ColumnSpacing(10);
            ColumnDefinition iconColumn;
            iconColumn.Width(GridLengthHelper::Auto());
            ColumnDefinition textColumn;
            textColumn.Width(GridLengthHelper::FromValueAndType(1, GridUnitType::Star));
            row.ColumnDefinitions().Append(iconColumn);
            row.ColumnDefinitions().Append(textColumn);
            auto icon = ui::Icon(glyph, 14);
            Microsoft::UI::Xaml::Automation::AutomationProperties::SetAccessibilityView(
                icon, Microsoft::UI::Xaml::Automation::Peers::AccessibilityView::Raw);
            icon.VerticalAlignment(VerticalAlignment::Top);
            icon.Margin(ThicknessHelper::FromLengths(0, 3, 0, 0));
            row.Children().Append(icon);
            StackPanel text;
            text.Children().Append(ui::Text(hstring(stateLabel) + L": " + title, L"BodyStrongTextBlockStyle"));
            if (!detail.empty()) text.Children().Append(ui::Text(detail, L"CaptionTextBlockStyle", 0.75));
            Grid::SetColumn(text, 1);
            row.Children().Append(text);
            return row;
        }
    }

    TrainPage::TrainPage()
    {
        InitializeComponent();
    }

    // ---- page lifetime ------------------------------------------------------------------------------------
    void TrainPage::OnNavigatedTo(NavigationEventArgs const&)
    {
        auto weak = get_weak();
        auto dispatcher = DispatcherQueue();
        m_projectToken = aiwf::AppState::Get().ProjectChanged([weak]
        {
            if (auto self = weak.get(); self && !aiwf::AppState::Get().ShuttingDown()) self->UpdateProjectState();
        });
        UpdateProjectState();
        m_statusToken = aiwf::EngineSupervisor::Instance().StatusChanged([weak, dispatcher]
        {
            dispatcher.TryEnqueue([weak]
            {
                if (auto self = weak.get(); self && !aiwf::AppState::Get().ShuttingDown()) self->UpdateEngineState();
            });
        });
        UpdateEngineState();
        // a package chosen on the Datasets page arrives here with its exact revision
        if (m_loaded) LoadPackagesAsync();
    }

    void TrainPage::OnNavigatedFrom(NavigationEventArgs const&)
    {
        aiwf::EngineSupervisor::Instance().StatusChanged(m_statusToken);
        aiwf::AppState::Get().ProjectChanged(m_projectToken);
        ++m_projectGeneration;
    }

    void TrainPage::UpdateProjectState()
    {
        auto projectId = aiwf::AppState::Get().ProjectId();
        if (m_projectId == projectId) return;
        m_projectId = std::move(projectId);
        ++m_projectGeneration;
        m_datasetId.clear();
        m_datasetSha.clear();
        m_datasetName.clear();
        ImportResult().Text(L"");
        PlanResult().Children().Clear();
        UpdateEngineState();
    }

    void TrainPage::UpdateEngineState()
    {
        bool ready = ui::UpdateEngineBar(EngineBar(), { L"engine-api", L"training", L"datasets" });
        if (ready && !m_loaded)
        {
            m_loaded = true;
            LoadModelsAsync(false);
            LoadPackagesAsync();
        }
        else if (!ready && !m_loaded && Models().Children().Size() == 0)
        {
            // say why the lists are empty: unavailable, not "you have no models"
            Models().Children().Append(ui::Text(L"Base models are listed once the engine API and the training engine are running (see above).",
                                                L"BodyTextBlockStyle", 0.75));
            PackagePicker().PlaceholderText(L"Packages are listed once the dataset engine is running");
        }
        ImportButton().IsEnabled(ready && !m_importing && !m_planning);
        PlanButton().IsEnabled(ready && !m_importing && !m_planning && !m_datasetId.empty() && !m_projectId.empty());
    }

    void TrainPage::StartEngine_Click(IInspectable const&, RoutedEventArgs const&)
    {
        ui::StartEnginesAsync({ L"engine-api", L"training", L"datasets" });
    }

    // ---- step 1: base models -----------------------------------------------------------------------------------
    fire_and_forget TrainPage::LoadModelsAsync(bool refresh)
    {
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        JsonObject body{ nullptr };
        hstring failure;
        try
        {
            body = co_await http::GetJsonAsync(aiwf::AppState::Get().EngineApi() + (refresh ? L"/models?refresh=true" : L"/models"));
        }
        catch (hresult_error const& error)
        {
            failure = error.message();
        }
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile
        if (!failure.empty())
        {
            Models().Children().Clear();
            Models().Children().Append(ui::Text(L"Could not list models: " + failure, L"BodyTextBlockStyle", 0.8));
            m_loaded = false;
            co_return;
        }
        m_hasKey = http::Bool(body, L"has_key");
        m_models.clear();
        // this loop keeps the display fields of each model the training engine offers
        for (auto const& value : http::Arr(body, L"models"))
        {
            auto item = value.GetObject();
            Model model;
            model.id = http::Str(item, L"model_id");
            model.label = http::Str(item, L"label", model.id.c_str());
            model.repo = http::Str(item, L"hf_repo");
            model.sizeB = http::Num(item, L"size_b");
            model.present = http::Bool(item, L"present");
            model.textCapable = http::Bool(item, L"text_capable", true);
            auto access = http::Obj(item, L"access");
            model.access = http::Str(access, L"state");
            model.downloadBytes = static_cast<uint64_t>(http::Num(access, L"download_bytes"));
            model.message = http::Str(access, L"message");
            m_models.push_back(std::move(model));
        }
        if (m_selectedModel.empty())
        {
            // default to the smallest text model already on this PC, which plans fastest
            for (auto const& model : m_models)
            {
                if (model.present && model.textCapable)
                {
                    m_selectedModel = model.id;
                    break;
                }
            }
        }
        ShowModels();
    }

    void TrainPage::ShowModels()
    {
        KeyState().Text(m_hasKey ? L"A Hugging Face key is saved on this PC; gated models you have accepted can be downloaded."
                                 : L"No Hugging Face key saved. It is only needed for gated models.");
        Models().Children().Clear();
        auto weak = get_weak();
        for (auto const& model : m_models)
        {
            Grid row;
            row.ColumnSpacing(12);
            // fixed columns so size, state chip and action line up from row to row
            for (auto width : { GridLengthHelper::FromValueAndType(1, GridUnitType::Star), GridLengthHelper::FromPixels(150),
                                GridLengthHelper::FromPixels(260), GridLengthHelper::FromPixels(150) })
            {
                ColumnDefinition definition;
                definition.Width(width);
                row.ColumnDefinitions().Append(definition);
            }

            RadioButton pick;
            pick.GroupName(L"base-model");
            pick.Content(box_value(hstring(model.label)));
            pick.IsChecked(model.id == m_selectedModel);
            pick.IsEnabled(model.textCapable);
            pick.Checked([weak, id = model.id](auto&&, auto&&)
            {
                if (auto self = weak.get(); self && !aiwf::AppState::Get().ShuttingDown()) self->m_selectedModel = id;
            });
            row.Children().Append(pick);

            auto size = ui::Text(hstring(SizeText(model.sizeB)), L"CaptionTextBlockStyle", 0.75);
            size.VerticalAlignment(VerticalAlignment::Center);
            Grid::SetColumn(size, 1);
            row.Children().Append(size);

            // this block turns weight availability into one honest label and at most one action
            hstring chip;
            uint32_t color = Neutral;
            Button action{ nullptr };
            if (!model.textCapable)
            {
                chip = L"Vision model: not for text packages";
            }
            else if (model.present)
            {
                chip = L"On this PC";
                color = Good;
            }
            else if (model.access == L"open" || model.access == L"gated_ok")
            {
                chip = hstring(L"Download " + (model.downloadBytes ? aiwf::paths::FormatBytes(model.downloadBytes) : std::wstring(L"available")));
                color = Caution;
                action = Button();
                action.Content(box_value(L"Download..."));
                action.Click([weak, model](auto&&, auto&&)
                {
                    if (auto self = weak.get(); self && !aiwf::AppState::Get().ShuttingDown()) self->DownloadAsync(model);
                });
            }
            else if (model.access == L"gated_no_key")
            {
                chip = L"Gated: needs your key";
                color = Caution;
                action = Button();
                action.Content(box_value(L"Add key..."));
                action.Click([weak](auto&&, auto&&) -> fire_and_forget
                {
                    auto self = weak.get();
                    if (!self) co_return;
                    if (co_await self->AskForKeyAsync(L"This model is gated on Hugging Face. Accept its license on huggingface.co with your account, then paste a read token here."))
                    {
                        self->LoadModelsAsync(true);
                    }
                });
            }
            else if (model.access == L"gated_no_access")
            {
                chip = L"Your key has no access yet";
                color = Bad;
                if (!model.repo.empty())
                {
                    action = Button();
                    action.Content(box_value(L"Open license page"));
                    action.Click([repo = model.repo](auto&&, auto&&) { ui::OpenWithShell(L"https://huggingface.co/" + repo); });
                }
            }
            else
            {
                chip = L"Not on this PC";
                if (!model.message.empty()) chip = hstring(L"Not on this PC (" + model.message + L")");
            }
            auto chipElement = ui::Chip(chip, color);
            chipElement.HorizontalAlignment(HorizontalAlignment::Left);
            Grid::SetColumn(chipElement, 2);
            row.Children().Append(chipElement);
            if (action)
            {
                action.HorizontalAlignment(HorizontalAlignment::Stretch);
                Grid::SetColumn(action, 3);
                row.Children().Append(action);
            }
            Models().Children().Append(row);
        }
    }

    fire_and_forget TrainPage::DownloadAsync(Model model)
    {
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        if (!m_downloadJob.empty()) co_return;   // one download at a time
        auto size = model.downloadBytes ? aiwf::paths::FormatBytes(model.downloadBytes) : std::wstring(L"an unknown amount");
        bool go = co_await ui::ConfirmAsync(XamlRoot(), hstring(L"Download " + model.label + L"?"),
                                            hstring(L"About " + size + L" from Hugging Face (" + model.repo + L"). It uses your network and disk space "
                                                    L"and can take a while. You can cancel at any time; partial files are kept so a later download resumes."),
                                            L"Download");
        if (!go) co_return;

        auto& state = aiwf::AppState::Get();
        JsonObject request;
        request.Insert(L"model_id", JsonValue::CreateStringValue(model.id));
        JsonObject started{ nullptr };
        hstring failure;
        int status = 0;
        try
        {
            started = co_await http::PostJsonAsync(state.EngineApi() + L"/models/download", request);
        }
        catch (hresult_error const& error)
        {
            failure = error.message();
            status = http::StatusOf(error);
        }
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile
        if (!failure.empty())
        {
            if (status == 403 && co_await AskForKeyAsync(L"Hugging Face needs your key for this model: " + failure))
            {
                LoadModelsAsync(true);
            }
            else if (status != 403)
            {
                co_await ui::ShowMessageAsync(XamlRoot(), L"Download not started", failure);
            }
            co_return;
        }

        auto job = http::Obj(started, L"download");
        m_downloadJob = http::Str(job, L"job_id");
        DownloadPanel().Visibility(Visibility::Visible);
        CancelDownloadButton().IsEnabled(true);
        auto jobUrl = state.EngineApi() + L"/models/downloads/" + http::EscapeSegment(hstring(m_downloadJob));
        std::wstring outcome;
        // this loop shows progress once a second until the download ends
        for (;;)
        {
            auto done = static_cast<uint64_t>(http::Num(job, L"done_bytes"));
            auto total = static_cast<uint64_t>(http::Num(job, L"total_bytes"));
            DownloadProgress().Value(total ? 100.0 * done / total : 0);
            DownloadText().Text(hstring(L"Downloading " + model.label + L": " + aiwf::paths::FormatBytes(done) + L" of " + aiwf::paths::FormatBytes(total) +
                                        L"  (" + std::to_wstring(static_cast<int>(http::Num(job, L"files_done"))) + L" of " +
                                        std::to_wstring(static_cast<int>(http::Num(job, L"files_total"))) + L" files)"));
            outcome = std::wstring(http::Str(job, L"status"));
            if (outcome != L"running" && outcome != L"queued" && outcome != L"starting") break;
            co_await resume_after(std::chrono::seconds(1));
            failure = L"";
            try
            {
                job = http::Obj(co_await http::GetJsonAsync(jobUrl), L"download");
            }
            catch (hresult_error const& error)
            {
                failure = error.message();
            }
            co_await wil::resume_foreground(dispatcher);
            if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile
            if (!failure.empty())
            {
                outcome = L"lost";
                break;
            }
        }
        m_downloadJob.clear();
        CancelDownloadButton().IsEnabled(false);
        if (outcome == L"completed")
        {
            DownloadText().Text(hstring(model.label + L" is on this PC now."));
            DownloadProgress().Value(100);
            LoadModelsAsync(false);
        }
        else if (outcome == L"cancelled")
        {
            DownloadText().Text(L"Download cancelled. Partial files are kept, so a later download resumes.");
        }
        else
        {
            auto message = http::Str(job, L"message");
            DownloadText().Text(L"The download stopped: " + (failure.empty() ? (message.empty() ? hstring(L"no reason given") : message) : failure));
        }
    }

    fire_and_forget TrainPage::CancelDownload_Click(IInspectable const&, RoutedEventArgs const&)
    {
        if (m_downloadJob.empty()) co_return;
        auto strong = get_strong();
        CancelDownloadButton().IsEnabled(false);
        try
        {
            co_await http::PostJsonAsync(aiwf::AppState::Get().EngineApi() + L"/models/downloads/" + http::EscapeSegment(hstring(m_downloadJob)) + L"/cancel",
                                         JsonObject());
        }
        catch (hresult_error const&)
        {
            // the progress loop reports the final state either way
        }
    }

    // ---- the Hugging Face key: entered here, checked by the training engine, never shown again ---------------
    fire_and_forget TrainPage::Key_Click(IInspectable const&, RoutedEventArgs const&)
    {
        auto strong = get_strong();
        if (co_await AskForKeyAsync(L"Paste a Hugging Face read token. It is checked with Hugging Face and saved in Hugging Face's standard token file on this PC."))
        {
            LoadModelsAsync(true);
        }
    }

    IAsyncOperation<bool> TrainPage::AskForKeyAsync(hstring reason)
    {
        // the dialog itself is shared with Setup (Services/Ui.cpp)
        auto strong = get_strong();
        co_return co_await ui::AskForHfKeyAsync(XamlRoot(), reason, m_hasKey);
    }

    // ---- step 2: training data ------------------------------------------------------------------------------------
    fire_and_forget TrainPage::LoadPackagesAsync()
    {
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        auto pending = aiwf::AppState::Get().TakePendingPackage();
        JsonObject body{ nullptr };
        hstring failure;
        try
        {
            body = co_await http::GetJsonAsync(aiwf::AppState::Get().EngineApi() + L"/datasets/packages");
        }
        catch (hresult_error const& error)
        {
            failure = error.message();
        }
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile
        if (!failure.empty())
        {
            PackagePicker().PlaceholderText(L"Packages unavailable: " + failure);
            co_return;
        }
        auto previous = ui::SelectedTag(PackagePicker());
        PackagePicker().Items().Clear();
        // this loop offers ready revisions only; the tag carries name and hash together
        for (auto const& value : http::Arr(body, L"packages"))
        {
            auto package = value.GetObject();
            if (http::Str(package, L"status") != L"ready") continue;
            auto name = std::wstring(http::Str(package, L"package_name"));
            auto sha = std::wstring(http::Str(package, L"manifest_sha256"));
            auto counts = http::Obj(package, L"counts");
            ComboBoxItem item;
            item.Content(box_value(hstring(name + L"   ·   revision " + sha.substr(0, 12) + L"   ·   " +
                                           std::to_wstring(static_cast<int>(http::Num(counts, L"train") + http::Num(counts, L"validation"))) + L" rows")));
            item.Tag(box_value(hstring(sha + L"|" + name)));
            PackagePicker().Items().Append(item);
            bool wanted = pending ? (pending->manifestSha256 == sha && pending->name == name) : (hstring(sha + L"|" + name) == previous);
            if (wanted) PackagePicker().SelectedItem(item);
        }
        // the placeholder may still carry an older "engine not running" message; say what is true now
        PackagePicker().PlaceholderText(PackagePicker().Items().Size() == 0 ? L"No published packages yet (see Datasets)"
                                                                            : L"Choose a package revision");
        if (pending && !PackagePicker().SelectedItem())
        {
            ImportResult().Text(hstring(L"The revision picked on Datasets (" + pending->manifestSha256.substr(0, 12) + L") is no longer published. Pick a current one."));
        }
    }

    fire_and_forget TrainPage::Import_Click(IInspectable const&, RoutedEventArgs const&)
    {
        if (m_importing || m_planning) co_return;
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        auto& state = aiwf::AppState::Get();
        auto tag = std::wstring(ui::SelectedTag(PackagePicker()));
        auto bar = tag.find(L'|');
        if (bar == std::wstring::npos)
        {
            ImportResult().Text(L"Choose a package revision first.");
            co_return;
        }
        if (state.ProjectId().empty())
        {
            ImportResult().Text(L"Choose a project in the title bar first; the import is recorded against it.");
            co_return;
        }
        auto sha = tag.substr(0, bar);
        auto name = tag.substr(bar + 1);
        auto generation = m_projectGeneration;
        JsonObject body;
        body.Insert(L"project_id", JsonValue::CreateStringValue(state.ProjectId()));
        body.Insert(L"package_name", JsonValue::CreateStringValue(name));
        body.Insert(L"manifest_sha256", JsonValue::CreateStringValue(sha));
        m_importing = true;
        UpdateEngineState();
        ImportResult().Text(L"Importing the exact revision into the training engine...");

        JsonObject result{ nullptr };
        hstring failure;
        int status = 0;
        try
        {
            result = co_await http::PostJsonAsync(state.EngineApi() + L"/retrain/import", body);
        }
        catch (hresult_error const& error)
        {
            failure = error.message();
            status = http::StatusOf(error);
        }
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile
        m_importing = false;
        UpdateEngineState();
        if (generation != m_projectGeneration) co_return;
        if (!failure.empty())
        {
            ImportResult().Text(status == 409 ? L"This package changed after you picked it, so nothing was imported. Refresh and pick the new revision."
                                              : L"Not imported: " + failure);
            m_datasetId.clear();
            UpdateEngineState();
            if (status == 409) LoadPackagesAsync();
            co_return;
        }
        auto dataset = http::Obj(result, L"dataset");
        auto counts = http::Obj(dataset, L"counts");
        m_datasetId = http::Str(dataset, L"dataset_id");
        m_datasetSha = sha;
        m_datasetName = name;
        UpdateEngineState();
        ImportResult().Text(hstring(L"Imported “" + name + L"” revision " + sha.substr(0, 12) + L": " +
                                    std::to_wstring(static_cast<int>(http::Num(counts, L"train"))) + L" train / " +
                                    std::to_wstring(static_cast<int>(http::Num(counts, L"validation"))) + L" validation rows" +
                                    (http::Bool(dataset, L"reused") ? L" (already in the training engine; reused)." : L".")));
    }

    // ---- step 3: the dry-run plan -----------------------------------------------------------------------------------
    fire_and_forget TrainPage::Plan_Click(IInspectable const&, RoutedEventArgs const&)
    {
        if (m_importing || m_planning) co_return;
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        auto& state = aiwf::AppState::Get();
        UpdateProjectState();
        auto generation = m_projectGeneration;
        auto panel = PlanResult();
        panel.Children().Clear();
        if (m_datasetId.empty() || m_selectedModel.empty() || state.ProjectId().empty())
        {
            panel.Children().Append(ui::Text(L"Pick a base model, import a package revision (step 2), and choose a project first.", L"BodyTextBlockStyle", 0.8));
            co_return;
        }
        JsonObject settings;
        settings.Insert(L"method", JsonValue::CreateStringValue(ui::SelectedTag(MethodPicker())));
        JsonObject body;
        body.Insert(L"project_id", JsonValue::CreateStringValue(state.ProjectId()));
        body.Insert(L"dataset_id", JsonValue::CreateStringValue(m_datasetId));
        body.Insert(L"manifest_sha256", JsonValue::CreateStringValue(m_datasetSha));
        body.Insert(L"model_id", JsonValue::CreateStringValue(m_selectedModel));
        body.Insert(L"settings", settings);
        m_planning = true;
        UpdateEngineState();
        ProgressRing ring;
        ring.IsActive(true);
        ring.Width(24);
        ring.Height(24);
        ring.HorizontalAlignment(HorizontalAlignment::Left);
        panel.Children().Append(ring);

        JsonObject response{ nullptr };
        hstring failure;
        try
        {
            response = co_await http::PostJsonAsync(state.EngineApi() + L"/retrain/preflight", body);
        }
        catch (hresult_error const& error)
        {
            failure = error.message();
        }
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile
        m_planning = false;
        UpdateEngineState();
        if (generation != m_projectGeneration) co_return;
        panel.Children().Clear();
        if (!failure.empty())
        {
            panel.Children().Append(ui::Text(L"The plan could not be checked: " + failure));
            co_return;
        }

        // this block shows the verdict, then the memory estimate, gates, dependencies and notes
        auto result = http::Obj(response, L"result");
        auto planStatus = std::wstring(http::Str(result, L"plan_status"));
        auto estimate = http::Obj(result, L"estimate");
        auto fit = std::wstring(http::Str(estimate, L"fit_state"));
        StackPanel verdict;
        verdict.Orientation(Orientation::Horizontal);
        verdict.Spacing(10);
        verdict.Children().Append(ui::Chip(hstring(L"Plan: " + planStatus), planStatus == L"ready" ? Good : Caution));
        if (estimate)
        {
            wchar_t memory[160]{};
            swprintf_s(memory, L"Needs about %.1f GB of %.1f GB video memory (%s)", http::Num(estimate, L"estimated_gb"), http::Num(estimate, L"limit_gb"),
                       fit.c_str());
            verdict.Children().Append(ui::Chip(memory, fit == L"safe" ? Good : (fit == L"tight" ? Caution : Bad)));
        }
        panel.Children().Append(verdict);

        panel.Children().Append(ui::Text(L"Checks", L"BodyStrongTextBlockStyle"));
        for (auto const& value : http::Arr(result, L"gates"))
        {
            auto gate = value.GetObject();
            panel.Children().Append(CheckRow(std::wstring(http::Str(gate, L"state")), http::Str(gate, L"gate"), http::Str(gate, L"detail")));
        }
        auto dependencies = http::Arr(result, L"dependencies");
        if (dependencies.Size() > 0)
        {
            panel.Children().Append(ui::Text(L"Software on this PC", L"BodyStrongTextBlockStyle"));
            for (auto const& value : dependencies)
            {
                auto dependency = value.GetObject();
                panel.Children().Append(CheckRow(http::Bool(dependency, L"available") ? L"ready" : L"missing", http::Str(dependency, L"label"), L""));
            }
        }
        for (auto const& note : http::Arr(result, L"notes"))
        {
            if (note.ValueType() == JsonValueType::String) panel.Children().Append(ui::Text(L"• " + note.GetString(), L"CaptionTextBlockStyle", 0.75));
        }
        panel.Children().Append(ui::Text(http::Str(response, L"message", L"Dry run only. Training was not started.") +
                                             L" Start the run in ReTrain when the checks pass.",
                                         L"CaptionTextBlockStyle", 0.7));
    }
}
