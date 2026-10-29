// Setup page implementation (see SetupPage.xaml.h for where each answer is saved).
#include "pch.h"
#include <cwctype>
#include <shellapi.h>
#include "SetupPage.xaml.h"
#if __has_include("SetupPage.g.cpp")
#include "SetupPage.g.cpp"
#endif
#include "Services/AppState.h"
#include "Services/EngineSupervisor.h"
#include "Services/GpuMonitor.h"
#include "Services/HardwareAdapters.h"
#include "Services/Http.h"
#include "Services/Paths.h"
#include "Services/Ui.h"
#include "Services/Credentials.h"

using namespace winrt;
using namespace winrt::Windows::Foundation;
using namespace winrt::Windows::Data::Json;
using namespace winrt::Microsoft::UI::Xaml;
using namespace winrt::Microsoft::UI::Xaml::Controls;
using namespace winrt::Microsoft::UI::Xaml::Navigation;
namespace ui = aiwf::ui;
namespace http = aiwf::http;
namespace fs = std::filesystem;

namespace winrt::AiwfDesktop::implementation
{
    namespace
    {
        // ---- the colors and glyphs of a check result -----------------------------------------------
        constexpr uint32_t Good = 0xFF2DA44E;
        constexpr uint32_t Caution = 0xFFE3A21A;
        constexpr uint32_t Bad = 0xFFD13438;
        constexpr wchar_t GlyphOk[] = L"\xE73E";        // check mark
        constexpr wchar_t GlyphWarn[] = L"\xE7BA";      // warning
        constexpr wchar_t GlyphInfo[] = L"\xE946";      // info

        constexpr wchar_t const* StepNames[] = { L"This PC", L"Save locations", L"Model locations", L"Engines", L"Review and apply" };

        // trims spaces and the quotes Explorer's "Copy as path" adds
        std::wstring CleanPath(std::wstring text)
        {
            auto isJunk = [](wchar_t ch) { return iswspace(ch) || ch == L'"'; };
            while (!text.empty() && isJunk(text.back())) text.pop_back();
            size_t start = 0;
            while (start < text.size() && isJunk(text[start])) ++start;
            return text.substr(start);
        }

        // one folder per line, cleaned, empty lines dropped
        std::vector<std::wstring> Lines(std::wstring const& text)
        {
            std::vector<std::wstring> lines;
            std::wstringstream stream(text);
            std::wstring line;
            while (std::getline(stream, line))
            {
                auto clean = CleanPath(line);
                if (!clean.empty()) lines.push_back(clean);
            }
            return lines;
        }

        std::wstring Join(std::vector<std::wstring> const& parts, wchar_t separator)
        {
            std::wstring out;
            for (auto const& part : parts)
            {
                if (!out.empty()) out.push_back(separator);
                out += part;
            }
            return out;
        }

        // "D:\a;E:\b" (an engine's environment) -> "D:\a\nE:\b" (one per line in the box)
        std::wstring SplitToLines(std::wstring const& text, wchar_t separator)
        {
            std::vector<std::wstring> parts;
            size_t start = 0;
            while (start <= text.size())
            {
                auto end = text.find(separator, start);
                if (end == std::wstring::npos) end = text.size();
                auto part = CleanPath(text.substr(start, end - start));
                if (!part.empty()) parts.push_back(part);
                start = end + 1;
            }
            return Join(parts, L'\n');
        }

        // a full path such as D:\Models or \\server\share\models (not "models" or "D:models")
        bool IsFullPath(std::wstring const& text)
        {
            return fs::path(text).is_absolute();
        }

        bool FolderExists(std::wstring const& text)
        {
            std::error_code ignored;
            return fs::is_directory(fs::path(text), ignored);
        }

        // free space on the drive holding path (or its nearest existing parent)
        std::optional<uint64_t> FreeBytes(fs::path path)
        {
            std::error_code ignored;
            while (!path.empty() && !fs::exists(path, ignored) && path.has_parent_path() && path.parent_path() != path) path = path.parent_path();
            ULARGE_INTEGER free{};
            if (path.empty() || !GetDiskFreeSpaceExW(path.c_str(), &free, nullptr, nullptr)) return std::nullopt;
            return free.QuadPart;
        }

        // a check row: colored glyph, title and detail, optional button at the end
        Grid CheckRow(uint32_t color, wchar_t const* glyph, hstring const& title, hstring const& detail, Button const& action = nullptr)
        {
            Grid row;
            row.ColumnSpacing(12);
            ColumnDefinition iconColumn;
            iconColumn.Width(GridLengthHelper::Auto());
            ColumnDefinition textColumn;
            textColumn.Width(GridLengthHelper::FromValueAndType(1, GridUnitType::Star));
            ColumnDefinition buttonColumn;
            buttonColumn.Width(GridLengthHelper::Auto());
            row.ColumnDefinitions().Append(iconColumn);
            row.ColumnDefinitions().Append(textColumn);
            row.ColumnDefinitions().Append(buttonColumn);

            auto icon = ui::Icon(glyph, 16);
            icon.Foreground(ui::ColorBrush(color));
            icon.VerticalAlignment(VerticalAlignment::Top);
            icon.Margin(ThicknessHelper::FromLengths(0, 3, 0, 0));
            row.Children().Append(icon);

            StackPanel text;
            text.Spacing(2);
            text.Children().Append(ui::Text(title, L"BodyStrongTextBlockStyle"));
            auto detailText = ui::Text(detail, L"CaptionTextBlockStyle", 0.75);
            detailText.IsTextSelectionEnabled(true);
            text.Children().Append(detailText);
            Grid::SetColumn(text, 1);
            row.Children().Append(text);

            if (action)
            {
                action.VerticalAlignment(VerticalAlignment::Center);
                Grid::SetColumn(action, 2);
                row.Children().Append(action);
            }
            return row;
        }

        Button TextButton(hstring const& label)
        {
            Button button;
            button.Content(box_value(label));
            return button;
        }
    }

    // ---- the questions this page asks ----------------------------------------------------------------------
    SetupPage::SetupPage()
    {
        InitializeComponent();

        // this list is every folder setup asks about, in the order the person sees them
        auto field = [](wchar_t const* key, wchar_t const* engineId, wchar_t const* title, wchar_t const* help,
                        bool list, wchar_t separator, int step)
        {
            FolderField made;
            made.key = key;
            made.engineId = engineId;
            made.title = title;
            made.help = help;
            made.list = list;
            made.separator = separator;
            made.step = step;
            return made;
        };
        // step 2: where new work is saved
        m_fields.push_back(field(L"output_dir", L"", L"Generated images",
                                 L"Create saves every image here. The Datasets page lists them for cataloging.", false, L'\n', 1));
        m_fields.push_back(field(L"DATASET_STUDIO_STATE", L"datasets", L"Dataset catalog and packages",
                                 L"Dataset Studio's database, collections and the training packages it publishes.", false, L';', 1));
        m_fields.push_back(field(L"DATASET_STUDIO_ROOTS", L"datasets", L"Folders the dataset engine may import from",
                                 L"Dataset Studio reads images only inside these folders. Add the generated-images folder to catalog Studio's output. One folder per line.",
                                 true, L';', 1));
        // step 3: where models are found
        m_fields.push_back(field(L"models_dir", L"", L"Main model library",
                                 L"Studio looks for models here first and saves model downloads here.", false, L'\n', 2));
        m_fields.push_back(field(L"ckpt_dir", L"", L"Image checkpoints",
                                 L"Single-file image models (SD 1.5, SDXL, Flux and similar).", false, L'\n', 2));
        m_fields.push_back(field(L"extra_model_dirs", L"", L"More model libraries",
                                 L"Searched as well, never written to. One folder per line.", true, L'\n', 2));
        m_fields.push_back(field(L"extra_ckpt_dirs", L"", L"More checkpoint folders",
                                 L"Searched as well, never written to. One folder per line.", true, L'\n', 2));
        m_fields.push_back(field(L"RETRAIN_MODEL_ROOT", L"training", L"Training models",
                                 L"Base models for fine-tuning. Model downloads from the Train page are saved here.", false, L';', 2));
        m_fields.push_back(field(L"RETRAIN_MODEL_ROOTS", L"training", L"More training model folders",
                                 L"Searched as well for base models. One folder per line.", true, L';', 2));

        BuildFolderRows();
    }

    // ---- page lifetime -------------------------------------------------------------------------------------
    void SetupPage::OnNavigatedTo(NavigationEventArgs const&)
    {
        m_pageActive = true;
        auto weak = get_weak();
        auto dispatcher = DispatcherQueue();
        // engine changes refresh the check list and the engine rows, and load the folders once the API answers
        m_statusToken = aiwf::EngineSupervisor::Instance().StatusChanged([weak, dispatcher]
        {
            dispatcher.TryEnqueue([weak]
            {
                auto self = weak.get();
                if (!self || aiwf::AppState::Get().ShuttingDown()) return;
                if (self->m_step == 0) self->ShowPcChecks();
                if (self->m_step == 3)
                {
                    self->ShowEngines();
                    bool answering = aiwf::EngineSupervisor::Instance().IsAnswering(L"engine-api");
                    if (!answering)
                    {
                        ++self->m_backendReadinessGeneration;
                        self->m_backendReadinessLoading = false;
                        self->BackendReadinessText().Text(L"Start the AIWF engine API to check the selected backend.");
                        auto const* api = aiwf::EngineSupervisor::Instance().Find(L"engine-api");
                        bool canStart = api && !api->processes.empty();
                        auto apiState = aiwf::EngineSupervisor::Instance().Status(L"engine-api").state;
                        self->StartEngineApiButton().Visibility(canStart && apiState != aiwf::EngineState::Starting
                            ? Visibility::Visible : Visibility::Collapsed);
                        self->StartEngineApiButton().IsEnabled(canStart && apiState != aiwf::EngineState::Starting);
                        self->RefreshBackendReadinessButton().IsEnabled(false);
                    }
                    else if (!self->m_lastEngineApiAnswering && !self->m_backendReadinessLoading)
                    {
                        self->RefreshBackendReadiness();
                    }
                    self->m_lastEngineApiAnswering = answering;
                }
                if (!self->m_launchLoaded && aiwf::EngineSupervisor::Instance().IsAnswering(L"engine-api")) self->LoadLaunchFoldersAsync();
            });
        });
        m_applied = false;
        ++m_backendReadinessGeneration;
        m_backendReadinessLoading = false;
        m_lastEngineApiAnswering = aiwf::EngineSupervisor::Instance().IsAnswering(L"engine-api");
        auto enabledBackends = aiwf::AppState::Get().EnabledImageBackends();
        bool pythonEnabled = std::find(enabledBackends.begin(), enabledBackends.end(), L"python") != enabledBackends.end();
        bool cppEnabled = std::find(enabledBackends.begin(), enabledBackends.end(), L"cpp") != enabledBackends.end();
        int32_t choice = pythonEnabled ? 1 : 0;
        LegacyPythonOnlyNotice().Visibility(pythonEnabled && !cppEnabled ? Visibility::Visible : Visibility::Collapsed);
        m_syncingBackendChoice = true;
        ImageBackendChoice().SelectedIndex(choice);
        m_syncingBackendChoice = false;
        LoadEngineFolders();
        LoadLaunchFoldersAsync();
        ShowStep(0);
    }

    void SetupPage::OnNavigatedFrom(NavigationEventArgs const&)
    {
        m_pageActive = false;
        ++m_backendReadinessGeneration;
        m_backendReadinessLoading = false;
        aiwf::EngineSupervisor::Instance().StatusChanged(m_statusToken);
    }

    // ---- moving between steps ------------------------------------------------------------------------------
    void SetupPage::ShowStep(int step)
    {
        int previousStep = m_step;
        m_step = (std::clamp)(step, 0, StepCount - 1);
        if (previousStep == 3 && m_step != 3)
        {
            ++m_backendReadinessGeneration;
            m_backendReadinessLoading = false;
        }
        StackPanel panels[] = { StepPc(), StepSave(), StepModels(), StepEngines(), StepReview() };
        for (int index = 0; index < StepCount; ++index)
        {
            panels[index].Visibility(index == m_step ? Visibility::Visible : Visibility::Collapsed);
        }
        StepLabel().Text(hstring(L"Step " + std::to_wstring(m_step + 1) + L" of " + std::to_wstring(StepCount) + L"  ·  " + StepNames[m_step]));
        StepProgress().Value(m_step + 1);
        BackButton().IsEnabled(m_step > 0 && !m_applying);
        NextButton().Content(box_value(m_step < StepCount - 1 ? L"Next" : (m_applied ? L"Done" : L"Apply and finish")));
        SkipButton().Visibility(m_applied ? Visibility::Collapsed : Visibility::Visible);
        StepScroll().ChangeView(nullptr, IReference<double>{ 0.0 }, nullptr, true);

        // each step refreshes what it shows when it becomes visible
        if (m_step == 0) ShowPcChecks();
        if (m_step == 1 || m_step == 2)
        {
            for (auto& field : m_fields) UpdateFacts(field);
        }
        if (m_step == 3)
        {
            ShowEngines();
            RefreshBackendReadiness();
            LoadKeyStateAsync();
            AttachCheckpointDownloadAsync();
        }
        if (m_step == 4 && !m_applied)
        {
            ResultBar().IsOpen(false);
            BuildReview();
        }
    }

    void SetupPage::Back_Click(IInspectable const&, RoutedEventArgs const&)
    {
        // going back after applying means the person wants to change something: allow a second apply
        if (m_applied)
        {
            m_applied = false;
            HomeButton().Visibility(Visibility::Collapsed);
            ResultBar().IsOpen(false);
        }
        ShowStep(m_step - 1);
    }

    void SetupPage::Next_Click(IInspectable const&, RoutedEventArgs const&)
    {
        if (m_step < StepCount - 1)
        {
            ShowStep(m_step + 1);
            return;
        }
        if (m_applied)
        {
            aiwf::AppState::Get().RequestNavigation(L"home");
            return;
        }
        ApplyAsync();
    }

    void SetupPage::ImageBackendChoice_SelectionChanged(IInspectable const&, SelectionChangedEventArgs const&)
    {
        if (m_syncingBackendChoice) return;
        if (m_step == 3)
        {
            ++m_backendReadinessGeneration;
            m_backendReadinessLoading = false;
            ShowEngines();
            RefreshBackendReadiness();
        }
        if (m_step == 4 && !m_applied) BuildReview();
    }

    void SetupPage::RefreshBackendReadiness_Click(IInspectable const&, RoutedEventArgs const&)
    {
        RefreshBackendReadiness();
    }

    void SetupPage::StartEngineApi_Click(IInspectable const&, RoutedEventArgs const&)
    {
        auto& supervisor = aiwf::EngineSupervisor::Instance();
        auto const* api = supervisor.Find(L"engine-api");
        if (!api || api->processes.empty() || supervisor.IsAnswering(L"engine-api") ||
            supervisor.Status(L"engine-api").state == aiwf::EngineState::Starting) return;
        BackendReadinessText().Text(L"Starting the AIWF engine API...");
        StartEngineApiButton().IsEnabled(false);
        ui::StartEnginesAsync({ L"engine-api" });
    }

    void SetupPage::RefreshBackendReadiness()
    {
        if (m_step != 3) return;

        auto selected = SelectedImageBackends();
        StartEngineApiButton().Visibility(Visibility::Collapsed);
        if (selected.empty())
        {
            ++m_backendReadinessGeneration;
            m_backendReadinessLoading = false;
            BackendReadinessText().Text(L"Choose at least one image backend to check its readiness.");
            RefreshBackendReadinessButton().IsEnabled(false);
            return;
        }

        auto& supervisor = aiwf::EngineSupervisor::Instance();
        if (!supervisor.IsAnswering(L"engine-api"))
        {
            ++m_backendReadinessGeneration;
            m_backendReadinessLoading = false;
            BackendReadinessText().Text(supervisor.Status(L"engine-api").state == aiwf::EngineState::Starting
                ? L"The AIWF engine API is starting. Readiness will be checked when it responds."
                : L"Start the AIWF engine API to check the selected backend.");
            RefreshBackendReadinessButton().IsEnabled(false);
            auto const* api = supervisor.Find(L"engine-api");
            bool canStart = api && !api->processes.empty();
            auto apiState = supervisor.Status(L"engine-api").state;
            StartEngineApiButton().Visibility(canStart && apiState != aiwf::EngineState::Starting
                ? Visibility::Visible : Visibility::Collapsed);
            StartEngineApiButton().IsEnabled(canStart && apiState != aiwf::EngineState::Starting);
            return;
        }

        if (m_backendReadinessLoading) return;
        m_backendReadinessLoading = true;
        auto generation = ++m_backendReadinessGeneration;
        BackendReadinessText().Text(L"Checking selected backends and installed models...");
        RefreshBackendReadinessButton().IsEnabled(false);
        LoadBackendReadinessAsync(generation);
    }

    fire_and_forget SetupPage::LoadBackendReadinessAsync(uint64_t generation)
    {
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        auto selected = SelectedImageBackends();
        std::wstring result;
        hstring failure;
        try
        {
            auto const& state = aiwf::AppState::Get();
            std::wstring url = std::wstring(state.EngineApi()) + L"/images/backends";
            std::wstring separator = L"?";
            for (auto const& id : selected)
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
            for (auto const& id : selected)
            {
                JsonObject item{ nullptr };
                for (auto const& value : backends)
                {
                    if (value.ValueType() != JsonValueType::Object) continue;
                    auto candidate = value.GetObject();
                    if (http::Str(candidate, L"id") == id)
                    {
                        item = candidate;
                        break;
                    }
                }
                std::wstring label = id == L"python" ? L"AIWF Python (Diffusers)" : L"AIWF C++ (stable-diffusion.cpp)";
                if (!item)
                {
                    if (!result.empty()) result += L"\n\n";
                    result += label + L"\nNo readiness result was returned. Update the engine service and check again.";
                    continue;
                }

                auto discoveredLabel = http::Str(item, L"label");
                if (!discoveredLabel.empty()) label = std::wstring(discoveredLabel);
                bool enabled = http::Bool(item, L"enabled");
                bool available = http::Bool(item, L"available");
                auto reason = std::wstring(http::Str(item, L"reason"));
                if (!result.empty()) result += L"\n\n";
                result += label + (enabled && available ? L" — Files detected; generation not checked" : L" — Needs setup");

                std::vector<std::wstring> modelLabels;
                for (auto const& modelValue : http::Arr(item, L"models"))
                {
                    if (modelValue.ValueType() != JsonValueType::Object) continue;
                    auto model = modelValue.GetObject();
                    bool textToImage = false;
                    for (auto const& capability : http::Arr(model, L"capabilities"))
                    {
                        if (capability.ValueType() == JsonValueType::String && capability.GetString() == L"text_to_image")
                        {
                            textToImage = true;
                            break;
                        }
                    }
                    if (!textToImage) continue;
                    auto modelLabel = std::wstring(http::Str(model, L"label"));
                    if (!modelLabel.empty()) modelLabels.push_back(std::move(modelLabel));
                }
                if (!modelLabels.empty())
                {
                    result += L"\nModels: ";
                    for (size_t index = 0; index < modelLabels.size(); ++index)
                    {
                        if (index) result += L", ";
                        result += modelLabels[index];
                    }
                }
                if (!reason.empty()) result += L"\n" + reason;
            }
        }
        catch (hresult_error const& error)
        {
            failure = error.message();
        }

        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown() || generation != m_backendReadinessGeneration) co_return;
        m_backendReadinessLoading = false;
        RefreshBackendReadinessButton().IsEnabled(aiwf::EngineSupervisor::Instance().IsAnswering(L"engine-api"));
        if (!failure.empty())
        {
            BackendReadinessText().Text(hstring(L"Could not check backend readiness. " + std::wstring(failure)));
            co_return;
        }
        BackendReadinessText().Text(hstring(result.empty() ? L"No readiness result was returned. Check again." : result));
    }

    std::vector<std::wstring> SetupPage::SelectedImageBackends()
    {
        switch (ImageBackendChoice().SelectedIndex())
        {
        case 0: return { L"cpp" };
        case 1: return { L"python", L"cpp" };
        default: return {};
        }
    }

    std::wstring SetupPage::DescribeImageBackends(std::vector<std::wstring> const& backends) const
    {
        bool python = std::find(backends.begin(), backends.end(), L"python") != backends.end();
        bool cpp = std::find(backends.begin(), backends.end(), L"cpp") != backends.end();
        if (python && cpp) return L"AIWF Python (Diffusers) and AIWF C++ (stable-diffusion.cpp)";
        if (python) return L"AIWF Python (Diffusers)";
        if (cpp) return L"AIWF C++ (stable-diffusion.cpp)";
        return L"no image backends";
    }

    // skipping keeps every folder as it is; setup stays in Configure Studio
    void SetupPage::Skip_Click(IInspectable const&, RoutedEventArgs const&)
    {
        if (!aiwf::AppState::Get().MarkSetupCompleted())
        {
            ResultBar().Severity(InfoBarSeverity::Error);
            ResultBar().Title(L"Setup choice was not saved");
            ResultBar().Message(L"AES Studio could not write %LOCALAPPDATA%\\Programs\\AES Studio Data\\settings.json. Your previous settings are still active; check that the file and folder are writable, then try again.");
            ResultBar().IsOpen(true);
            return;
        }
        aiwf::AppState::Get().RequestNavigation(L"home");
    }

    void SetupPage::Home_Click(IInspectable const&, RoutedEventArgs const&)
    {
        aiwf::AppState::Get().RequestNavigation(L"home");
    }

    // ---- step 1: this PC -------------------------------------------------------------------------------------
    void SetupPage::ShowPcChecks()
    {
        PcRows().Children().Clear();
        auto weak = get_weak();

        // graphics card: chat, images and training all run on it
        auto gpu = aiwf::GpuMonitor::Instance().Sample();
        auto computeLevel = aiwf::hardware::ClassifyCudaComputeCapability(gpu.computeMajor, gpu.computeMinor);
        if (gpu.available)
        {
            std::wstring detail = gpu.name + L"  ·  " + aiwf::paths::FormatBytes(gpu.totalBytes) + L" video memory";
            if (!gpu.driver.empty()) detail += L"  ·  driver " + gpu.driver;
            if (!gpu.cuda.empty()) detail += L"  ·  driver supports CUDA " + gpu.cuda;
            if (computeLevel != aiwf::hardware::CudaComputeLevel::Unknown)
                detail += L"  ·  compute capability " + std::to_wstring(gpu.computeMajor) + L"." + std::to_wstring(gpu.computeMinor);
            PcRows().Children().Append(CheckRow(Good, GlyphOk, L"Graphics card", hstring(detail)));
            auto driverLevel = aiwf::hardware::ClassifyCudaDriver(gpu.cuda);
            if (driverLevel == aiwf::hardware::CudaDriverLevel::OlderThan13)
                PcRows().Children().Append(CheckRow(Caution, GlyphWarn, L"NVIDIA driver",
                    hstring(L"The driver reports CUDA " + gpu.cuda + L". The bundled image engine needs a CUDA 13 compatible NVIDIA driver; update the driver before trying it.")));
            else if (driverLevel == aiwf::hardware::CudaDriverLevel::Unknown)
                PcRows().Children().Append(CheckRow(Caution, GlyphWarn, L"NVIDIA driver",
                    L"The driver's CUDA support could not be determined. The bundled image engine needs a CUDA 13 compatible NVIDIA driver."));
            if (computeLevel == aiwf::hardware::CudaComputeLevel::OlderThan75)
                PcRows().Children().Append(CheckRow(Caution, GlyphWarn, L"NVIDIA GPU compatibility",
                    L"This GPU is below compute capability 7.5, the minimum for the bundled CUDA 13 image engine. A separately configured compatible runtime may still work."));
            else if (computeLevel == aiwf::hardware::CudaComputeLevel::Unknown)
                PcRows().Children().Append(CheckRow(Caution, GlyphWarn, L"NVIDIA GPU compatibility",
                    L"The GPU's compute capability could not be determined. The bundled CUDA 13 image engine requires compute capability 7.5 or later."));
        }
        else
        {
            auto adapters = aiwf::hardware::EnumerateDisplayAdapters();
            for (auto const& adapter : adapters)
                PcRows().Children().Append(CheckRow(Caution, GlyphInfo, L"Graphics card",
                    hstring(aiwf::hardware::DescribeDisplayAdapter(adapter) + L"  ·  reported by Windows")));
            if (adapters.empty())
                PcRows().Children().Append(CheckRow(Caution, GlyphWarn, L"Graphics card",
                    L"Windows did not report a hardware graphics adapter."));
            PcRows().Children().Append(CheckRow(Caution, GlyphWarn, L"NVIDIA image runtime",
                hstring(L"NVIDIA GPU telemetry is unavailable (" + gpu.error + L"). The bundled C++ image engine requires a supported NVIDIA GPU and a compatible CUDA 13 driver.")));
        }

        MEMORYSTATUSEX memory{}; memory.dwLength = sizeof(memory);
        bool memoryKnown = GlobalMemoryStatusEx(&memory) != FALSE;
        constexpr uint64_t minimumStartingRam = 12ULL * 1024 * 1024 * 1024;
        if (memoryKnown && memory.ullTotalPhys >= minimumStartingRam)
            PcRows().Children().Append(CheckRow(Good, GlyphInfo, L"System memory", hstring(aiwf::paths::FormatBytes(memory.ullTotalPhys))));
        else if (memoryKnown)
            PcRows().Children().Append(CheckRow(Caution, GlyphWarn, L"System memory",
                hstring(aiwf::paths::FormatBytes(memory.ullTotalPhys) + L". The SDXL starting suggestion uses a conservative 12 GiB system-memory threshold.")));
        else
            PcRows().Children().Append(CheckRow(Caution, GlyphWarn, L"System memory",
                L"Windows could not report system memory, so Studio cannot make its automatic SDXL suggestion."));
        SYSTEM_INFO cpu{}; GetNativeSystemInfo(&cpu);
        PcRows().Children().Append(CheckRow(Good, GlyphInfo, L"Processor", hstring(std::to_wstring(cpu.dwNumberOfProcessors) + L" logical processors")));
        std::error_code diskError;
        auto space = fs::space(aiwf::paths::DataDir(), diskError);
        if (!diskError) PcRows().Children().Append(CheckRow(Good, GlyphInfo, L"Free space on the profile drive", hstring(aiwf::paths::FormatBytes(space.available) + L" (a custom model folder may be on another drive)")));
        bool suggestSdxl = aiwf::hardware::MaySuggestBundledSdxl(gpu.available, gpu.totalBytes, gpu.cuda, gpu.computeMajor, gpu.computeMinor, memoryKnown ? memory.ullTotalPhys : 0);
        UseSuggestedSdxlButton().Visibility(suggestSdxl ? Visibility::Visible : Visibility::Collapsed);
        if (suggestSdxl)
        {
            auto modelPage = TextButton(L"Open model page");
            modelPage.Click([](auto&&, auto&&) { ui::OpenWithShell(L"https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors"); });
            PcRows().Children().Append(CheckRow(Caution, GlyphInfo, L"Suggested starting model: SDXL Base 1.0",
                L"6.94 GB download. Review the model license and download the safetensors checkpoint into your selected Image checkpoints folder. Actual fit depends on available system memory, free VRAM, driver and image settings. The bundled C++ runtime needs a compatible NVIDIA CUDA 13 driver.", modelPage));
        }
        else
        {
            PcRows().Children().Append(CheckRow(Caution, GlyphWarn, L"Starting model",
                L"No automatic recommendation for the bundled image engine on this PC. The SDXL suggestion requires a compatible NVIDIA GPU, CUDA 13 driver and at least 12 GiB system memory. Choose a compatible model and runtime manually."));
        }

        // the AES Studio code folder: the engine API and Pro run from it
        auto root = aiwf::paths::StudioRoot();
        bool rootFound = aiwf::paths::IsStudioRoot(root);
        auto choose = TextButton(L"Choose folder...");
        choose.Click([weak](auto&&, auto&&) { if (auto self = weak.get()) self->ChooseStudioRootAsync(); });
        PcRows().Children().Append(CheckRow(rootFound ? Good : Bad, rootFound ? GlyphOk : GlyphWarn, L"AES Studio folder",
            rootFound ? hstring(root.wstring())
                      : hstring(L"Not found next to this app. Choose the folder AES Studio was installed or cloned into (it contains aiwf\\engine_api.py)."),
            choose));

        // the Python environment used by the engine API and AIWF's own Diffusers backend
        auto python = root / L"venv" / L"Scripts" / L"python.exe";
        std::error_code ignored;
        bool pythonFound = rootFound && fs::exists(python, ignored);
        PcRows().Children().Append(CheckRow(pythonFound ? Good : Bad, pythonFound ? GlyphOk : GlyphWarn, L"Engine API Python environment",
            pythonFound ? hstring(python.wstring())
                        : hstring(L"Missing from the selected AES Studio folder. Choose a complete installation, or uninstall and reinstall AES Studio into a new empty folder. Your separate data folder, models and outputs are preserved. Python inference packages are separate.")));

        // the engine API: the settings below are read and saved through it
        auto& supervisor = aiwf::EngineSupervisor::Instance();
        auto status = supervisor.Status(L"engine-api");
        bool answering = supervisor.IsAnswering(L"engine-api");
        Button start{ nullptr };
        if (!answering && status.state != aiwf::EngineState::Starting && supervisor.Find(L"engine-api"))
        {
            start = TextButton(L"Start");
            start.Click([](auto&&, auto&&) { ui::StartEnginesAsync({ L"engine-api" }); });
        }
        std::wstring apiDetail = aiwf::StateLabel(status.state);
        if (!status.detail.empty() && status.state != aiwf::EngineState::Starting) apiDetail += L"  ·  " + status.detail;
        if (!answering) apiDetail += L"  ·  Needed to read and save the image and model folders.";
        PcRows().Children().Append(CheckRow(answering ? Good : (status.state == aiwf::EngineState::Starting ? Caution : Bad),
                                            answering ? GlyphOk : GlyphWarn, L"AES engine API", hstring(apiDetail), start));

        // where this app keeps its own settings and logs
        PcRows().Children().Append(CheckRow(Good, GlyphInfo, L"This app's settings and logs", hstring(aiwf::paths::DataDir().wstring())));
    }

    fire_and_forget SetupPage::ChooseStudioRootAsync()
    {
        auto strong = get_strong();
        auto picked = std::wstring(co_await ui::PickFolderAsync(XamlRoot()));
        if (picked.empty() || aiwf::AppState::Get().ShuttingDown()) co_return;
        if (!aiwf::paths::IsStudioRoot(picked))
        {
            co_await ui::ShowMessageAsync(XamlRoot(), L"That is not the AES Studio folder",
                hstring(picked + L" has no aiwf\\engine_api.py. Choose the folder AES Studio was installed or cloned into, for example the folder chosen in the AES Studio installer."));
            co_return;
        }
        std::wstring selectionProblem;
        if (!aiwf::paths::ValidateStudioRootSelection(std::filesystem::path(picked), selectionProblem))
        {
            co_await ui::ShowMessageAsync(XamlRoot(), L"This Studio folder cannot be selected", hstring(selectionProblem));
            co_return;
        }
        // saved right away: the engine API cannot start without it, and the later steps need the API
        if (!aiwf::AppState::Get().SetStudioRootSetting(picked))
        {
            co_await ui::ShowMessageAsync(XamlRoot(), L"Folder choice was not saved",
                L"AES Studio could not write %LOCALAPPDATA%\\Programs\\AES Studio Data\\settings.json. Your previous Studio folder remains active; check that the file and folder are writable, then try again.");
            co_return;
        }
        ShowPcChecks();
    }

    // ---- steps 2 and 3: the folder rows ------------------------------------------------------------------------
    void SetupPage::BuildFolderRows()
    {
        auto weak = get_weak();
        // this loop builds one card per folder question under its step
        for (size_t index = 0; index < m_fields.size(); ++index)
        {
            auto& field = m_fields[index];
            StackPanel content;
            content.Spacing(6);
            content.Children().Append(ui::Text(hstring(field.title), L"BodyStrongTextBlockStyle"));
            content.Children().Append(ui::Text(hstring(field.help), L"CaptionTextBlockStyle", 0.7));

            Grid entry;
            entry.ColumnSpacing(8);
            ColumnDefinition boxColumn;
            boxColumn.Width(GridLengthHelper::FromValueAndType(1, GridUnitType::Star));
            ColumnDefinition buttonColumn;
            buttonColumn.Width(GridLengthHelper::Auto());
            entry.ColumnDefinitions().Append(boxColumn);
            entry.ColumnDefinitions().Append(buttonColumn);

            TextBox box;
            Automation::AutomationProperties::SetName(box, hstring(field.title));
            if (field.list)
            {
                box.AcceptsReturn(true);
                box.TextWrapping(TextWrapping::Wrap);
                box.MinHeight(72);
            }
            box.TextChanged([weak, index](auto&&, auto&&)
            {
                if (auto self = weak.get(); self && !aiwf::AppState::Get().ShuttingDown()) self->UpdateFacts(self->m_fields[index]);
            });
            entry.Children().Append(box);

            auto browse = TextButton(field.list ? L"Add folder..." : L"Browse...");
            browse.VerticalAlignment(VerticalAlignment::Top);
            Automation::AutomationProperties::SetName(browse, hstring((field.list ? L"Add folder to " : L"Browse for ") + field.title));
            browse.Click([weak, index](auto&&, auto&&) { if (auto self = weak.get()) self->BrowseAsync(index); });
            Grid::SetColumn(browse, 1);
            entry.Children().Append(browse);
            content.Children().Append(entry);

            auto facts = ui::Text(L"", L"CaptionTextBlockStyle", 0.75);
            facts.IsTextSelectionEnabled(true);
            content.Children().Append(facts);

            field.box = box;
            field.facts = facts;
            (field.step == 1 ? SaveRows() : ModelRows()).Children().Append(ui::Card(content));
        }
    }

    fire_and_forget SetupPage::BrowseAsync(size_t index)
    {
        auto strong = get_strong();
        auto picked = std::wstring(co_await ui::PickFolderAsync(XamlRoot()));
        if (picked.empty() || aiwf::AppState::Get().ShuttingDown() || index >= m_fields.size()) co_return;
        auto& field = m_fields[index];
        if (!field.list)
        {
            field.box.Text(picked);
            co_return;
        }
        // lists gain a line, unless that folder is already there
        auto lines = Lines(std::wstring(field.box.Text()));
        bool present = std::any_of(lines.begin(), lines.end(), [&](std::wstring const& line) { return _wcsicmp(line.c_str(), picked.c_str()) == 0; });
        if (!present) lines.push_back(picked);
        field.box.Text(hstring(Join(lines, L'\n')));
    }

    std::wstring SetupPage::CurrentValue(FolderField const& field) const
    {
        if (!field.box) return field.original;
        auto text = std::wstring(field.box.Text());
        return field.list ? Join(Lines(text), L'\n') : CleanPath(text);
    }

    // the line under each box: what will actually be used, and whether it is there
    void SetupPage::UpdateFacts(FolderField& field)
    {
        if (!field.facts) return;
        if (!field.available)
        {
            field.facts.Text(field.engineId.empty()
                ? hstring(L"Waiting for the engine API" + (m_launchProblem.empty() ? std::wstring() : L": " + m_launchProblem) + L" (see step 1).")
                : hstring(L"AES Studio does not start this engine on this PC, so it keeps its own folders. Set up how it starts in the engine list (step 4) to choose them here."));
            return;
        }
        auto value = CurrentValue(field);
        if (field.list)
        {
            auto lines = Lines(value);
            if (lines.empty())
            {
                auto fallback = Lines(field.fallback);
                if (fallback.empty()) field.facts.Text(L"None added.");
                else if (field.engineId.empty()) field.facts.Text(hstring(L"None added. Studio also searches " + Join(fallback, L';') + L" on its own."));
                else field.facts.Text(hstring(L"Using " + field.fallbackFrom + L" (" + std::to_wstring(fallback.size()) +
                                              (fallback.size() == 1 ? L" folder)." : L" folders).")));
                return;
            }
            std::vector<std::wstring> problems;
            for (auto const& line : lines)
            {
                if (!IsFullPath(line)) problems.push_back(line + L" is not a full path");
                else if (!FolderExists(line)) problems.push_back(line + L" does not exist yet");
            }
            auto summary = std::to_wstring(lines.size()) + (lines.size() == 1 ? L" folder" : L" folders");
            field.facts.Text(hstring(problems.empty() ? summary + L", all found." : summary + L":\n" + Join(problems, L'\n')));
            return;
        }
        // facts about one folder: there or not, and the free space on its drive
        auto describe = [](std::wstring const& path)
        {
            auto free = FreeBytes(path);
            std::wstring freeText = free ? L"  ·  " + aiwf::paths::FormatBytes(*free) + L" free on that drive" : L"";
            return (FolderExists(path) ? std::wstring(L"found") : std::wstring(L"does not exist yet; it can be created when you apply")) + freeText;
        };
        if (value.empty())
        {
            // an empty box means the default (or the engine list's value); describe that folder instead
            if (field.fallback.empty()) field.facts.Text(L"Using the engine's own default folder.");
            else field.facts.Text(hstring(L"Using " + field.fallbackFrom + L": " + describe(field.fallback)));
            return;
        }
        if (!IsFullPath(value))
        {
            field.facts.Text(L"Use a full folder path such as D:\\AI\\Models.");
            return;
        }
        auto facts = describe(value);
        facts[0] = towupper(facts[0]);
        field.facts.Text(hstring(facts));
    }

    // Dataset Studio and ReTrain folders: the setup choice, else what the engine list sets, else the engine's default
    void SetupPage::LoadEngineFolders()
    {
        auto& supervisor = aiwf::EngineSupervisor::Instance();
        for (auto& field : m_fields)
        {
            if (field.engineId.empty()) continue;
            auto const* spec = supervisor.Find(field.engineId);
            field.available = spec && !spec->processes.empty();

            std::wstring chosen;
            for (auto const& [name, value] : aiwf::AppState::Get().EngineEnv(field.engineId))
            {
                if (_wcsicmp(name.c_str(), field.key.c_str()) == 0) chosen = value;
            }
            std::wstring fromList;
            if (field.available)
            {
                for (auto const& [name, value] : spec->processes.front().env)
                {
                    if (_wcsicmp(name.c_str(), field.key.c_str()) == 0) fromList = aiwf::paths::Expand(value);
                }
            }
            field.original = field.list ? SplitToLines(chosen, field.separator) : CleanPath(chosen);
            field.fallback = field.list ? SplitToLines(fromList, field.separator) : CleanPath(fromList);
            field.fallbackFrom = field.list ? L"the engine list's folders" : L"the engine list's folder";
            if (field.box)
            {
                field.box.Text(hstring(field.original));
                field.box.IsEnabled(field.available);
                field.box.PlaceholderText(hstring(field.fallback.empty() ? L"Engine default" : L"Engine list: " + field.fallback));
            }
            UpdateFacts(field);
        }
    }

    // image output and model folders: read from launch.json through the engine API
    fire_and_forget SetupPage::LoadLaunchFoldersAsync()
    {
        if (m_launchLoading || m_launchLoaded) co_return;
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        m_launchLoading = true;
        JsonObject body{ nullptr };
        hstring failure;
        if (aiwf::EngineSupervisor::Instance().IsAnswering(L"engine-api"))
        {
            try
            {
                body = co_await http::GetJsonAsync(aiwf::AppState::Get().EngineApi() + L"/setup");
            }
            catch (hresult_error const& error)
            {
                failure = error.message();
            }
        }
        else
        {
            failure = L"it is not running yet";
        }
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile
        m_launchLoading = false;
        m_launchProblem = std::wstring(failure);
        m_launchLoaded = failure.empty();

        auto folders = m_launchLoaded ? http::Obj(body, L"folders") : JsonObject{ nullptr };
        // this loop fills each launch.json folder box with its saved value and shows the default as a hint
        for (auto& field : m_fields)
        {
            if (!field.engineId.empty()) continue;
            field.available = m_launchLoaded;
            if (field.box) field.box.IsEnabled(m_launchLoaded);
            if (!m_launchLoaded)
            {
                UpdateFacts(field);
                continue;
            }
            auto info = http::Obj(folders, field.key.c_str());
            if (field.list)
            {
                std::vector<std::wstring> saved;
                for (auto const& item : http::Arr(info, L"saved"))
                {
                    if (item.ValueType() == JsonValueType::String) saved.emplace_back(item.GetString());
                }
                // folders found on their own are mentioned, not saved
                std::vector<std::wstring> effective;
                for (auto const& item : http::Arr(info, L"effective"))
                {
                    if (item.ValueType() == JsonValueType::Object) effective.emplace_back(http::Str(item.GetObject(), L"path"));
                }
                field.original = Join(saved, L'\n');
                // folders Studio adds by itself (effective but not saved)
                std::vector<std::wstring> automatic;
                for (auto const& path : effective)
                {
                    bool isSaved = std::any_of(saved.begin(), saved.end(), [&](std::wstring const& item) { return _wcsicmp(item.c_str(), path.c_str()) == 0; });
                    if (!isSaved) automatic.push_back(path);
                }
                field.fallback = Join(automatic, L'\n');
                field.fallbackFrom = L"folders Studio finds on its own";
                if (field.box) field.box.PlaceholderText(L"One folder per line");
            }
            else
            {
                field.original = CleanPath(std::wstring(http::Str(info, L"saved")));
                field.fallback = std::wstring(http::Str(info, L"default"));
                field.fallbackFrom = L"the default folder " + field.fallback;
                if (field.box) field.box.PlaceholderText(hstring(L"Default: " + field.fallback));
            }
            if (field.box) field.box.Text(hstring(field.original));
            UpdateFacts(field);
        }
    }

    // ---- step 4: engines and the Hugging Face key ---------------------------------------------------------------
    void SetupPage::ShowEngines()
    {
        EngineRows().Children().Clear();
        auto& supervisor = aiwf::EngineSupervisor::Instance();
        // this loop shows one card per engine: state, how it starts here, and a Start button when it can
        for (auto const& engine : supervisor.Engines())
        {
            if (engine.id == L"image") continue; // legacy external image service; AIWF image backends run through engine-api
            auto status = supervisor.Status(engine.id);
            bool answering = supervisor.IsAnswering(engine.id);
            std::wstring how;
            uint32_t color = answering ? Good : Caution;
            if (!engine.probeEnabled)
            {
                how = L"Disabled by default. Configure this service and set probe to true in the engine list to connect.";
            }
            else if (engine.processes.empty())
            {
                how = L"Not started from AES Studio on this PC; used when it is already running.";
            }
            else
            {
                auto const& first = engine.processes.front();
                auto exe = aiwf::paths::Expand(first.exe);
                // an engine started through a .cmd or .bat script names the script, not cmd.exe
                auto shown = exe;
                if (_wcsicmp(fs::path(exe).filename().c_str(), L"cmd.exe") == 0)
                {
                    for (auto const& arg : first.args)
                    {
                        auto extension = fs::path(arg).extension().wstring();
                        if (_wcsicmp(extension.c_str(), L".cmd") == 0 || _wcsicmp(extension.c_str(), L".bat") == 0) shown = aiwf::paths::Expand(arg);
                    }
                }
                std::error_code ignored;
                bool found = fs::exists(exe, ignored) && fs::exists(shown, ignored);
                how = L"Starts from " + shown + (found ? L"" : L". That file is missing, so starting will fail; fix the path in the engine list.");
                if (!found && !answering) color = Bad;
            }
            Button start{ nullptr };
            if (!answering && !engine.processes.empty() && status.state != aiwf::EngineState::Starting)
            {
                start = TextButton(L"Start");
                start.Click([id = engine.id](auto&&, auto&&) { ui::StartEnginesAsync({ id }); });
            }
            std::wstring detail = engine.role + L"\n" + how;
            if (!status.detail.empty() && !answering) detail += L"\n" + status.detail;
            EngineRows().Children().Append(ui::Card(CheckRow(color, answering ? GlyphOk : GlyphWarn,
                                                             hstring(engine.name + L"  ·  " + aiwf::StateLabel(status.state)), hstring(detail), start)));
        }

        KeyButton().IsEnabled(true);
        LoadKeyStateAsync();
    }

    fire_and_forget SetupPage::LoadKeyStateAsync()
    {
        m_hasKey = aiwf::credentials::Has(L"Hugging Face");
        m_hasCivitaiKey = aiwf::credentials::Has(L"Civitai");
        KeyText().Text(m_hasKey ? L"A key is saved in Windows Credential Manager. It has not been validated here."
                                : L"No key saved. Public models can be used without one.");
        CivitaiKeyText().Text(m_hasCivitaiKey ? L"A key is saved in Windows Credential Manager. It has not been validated here."
                                             : L"No key saved. You can add one now or later.");
        co_return;
    }

    fire_and_forget SetupPage::Key_Click(IInspectable const&, RoutedEventArgs const&)
    {
        auto strong = get_strong();
        if (co_await ui::AskForProviderKeyAsync(XamlRoot(), L"Hugging Face", m_hasKey)) LoadKeyStateAsync();
    }

    fire_and_forget SetupPage::CivitaiKey_Click(IInspectable const&, RoutedEventArgs const&)
    {
        auto strong = get_strong();
        if (co_await ui::AskForProviderKeyAsync(XamlRoot(), L"Civitai", m_hasCivitaiKey)) LoadKeyStateAsync();
    }

    void SetupPage::UseSuggestedSdxl_Click(IInspectable const&, RoutedEventArgs const&)
    {
        CheckpointSource().SelectedIndex(0);
        CheckpointReference().Text(L"stabilityai/stable-diffusion-xl-base-1.0");
        CheckpointFilename().Text(L"sd_xl_base_1.0.safetensors");
    }

    fire_and_forget SetupPage::CheckpointDownload_Click(IInspectable const&, RoutedEventArgs const&)
    {
        auto strong = get_strong();
        if (m_checkpointDownloading) co_return;
        if (!aiwf::EngineSupervisor::Instance().IsAnswering(L"engine-api"))
        {
            co_await ui::ShowMessageAsync(XamlRoot(), L"Engine API is not ready", L"Start the local engine API before downloading a checkpoint.");
            co_return;
        }
        auto source = CheckpointSource().SelectedIndex() == 1 ? L"civitai" : L"huggingface";
        auto reference = CheckpointReference().Text();
        auto filename = CheckpointFilename().Text();
        if (reference.empty() || filename.empty())
        {
            co_await ui::ShowMessageAsync(XamlRoot(), L"Enter the model", L"Enter the provider reference and model filename first.");
            co_return;
        }
        auto extension = fs::path(std::wstring(filename)).extension().wstring();
        std::transform(extension.begin(), extension.end(), extension.begin(), [](wchar_t c) { return static_cast<wchar_t>(std::towlower(c)); });
        bool legacy = extension == L".ckpt" || extension == L".pt" || extension == L".pth";
        std::wstring message = L"Model files can be several gigabytes. Review the model source and license. The file goes to the currently saved Image checkpoints folder and will not overwrite an existing file.";
        if (legacy) message += L" Legacy PyTorch files can contain executable code when loaded. Only use a model you trust.";
        if (!co_await ui::ConfirmAsync(XamlRoot(), L"Download checkpoint?", hstring(message), L"Download", L"Cancel")) co_return;
        if (aiwf::AppState::Get().ShuttingDown()) co_return;
        JsonObject body;
        body.Insert(L"source", JsonValue::CreateStringValue(source));
        body.Insert(L"reference", JsonValue::CreateStringValue(reference));
        body.Insert(L"filename", JsonValue::CreateStringValue(filename));
        JsonObject result{ nullptr }; hstring failure;
        try { result = co_await http::PostJsonAsync(aiwf::AppState::Get().EngineApi() + L"/checkpoint-downloads", body); }
        catch (hresult_error const& error) { failure = error.message(); }
        if (!failure.empty())
        {
            co_await ui::ShowMessageAsync(XamlRoot(), L"Download could not start", failure);
            co_return;
        }
        auto id = std::wstring(http::Str(result, L"id"));
        if (id.empty())
        {
            co_await ui::ShowMessageAsync(XamlRoot(), L"Download could not start", L"The engine did not return a download ID.");
            co_return;
        }
        TrackCheckpointDownloadAsync(id);
    }

    fire_and_forget SetupPage::AttachCheckpointDownloadAsync()
    {
        if (m_checkpointAttaching || m_checkpointMonitoring || !aiwf::EngineSupervisor::Instance().IsAnswering(L"engine-api")) co_return;
        m_checkpointAttaching = true;
        auto strong = get_strong();
        JsonObject state{ nullptr };
        try { state = co_await http::GetJsonAsync(aiwf::AppState::Get().EngineApi() + L"/checkpoint-downloads"); }
        catch (hresult_error const&) {}
        m_checkpointAttaching = false;
        if (!m_pageActive || aiwf::AppState::Get().ShuttingDown()) co_return;
        auto job = http::Obj(state, L"job");
        auto id = std::wstring(http::Str(job, L"id"));
        if (!id.empty() && !m_checkpointMonitoring) TrackCheckpointDownloadAsync(id);
    }

    fire_and_forget SetupPage::TrackCheckpointDownloadAsync(std::wstring id)
    {
        if (m_checkpointMonitoring) co_return;
        auto strong = get_strong();
        m_checkpointDownloadId = std::move(id);
        m_checkpointDownloading = true;
        CheckpointDownloadButton().IsEnabled(false);
        CheckpointCancelButton().IsEnabled(true);
        auto dispatcher = DispatcherQueue();
        m_checkpointMonitoring = true;
        while (m_pageActive && !aiwf::AppState::Get().ShuttingDown())
        {
            co_await resume_after(std::chrono::seconds(1));
            co_await wil::resume_foreground(dispatcher);
            if (!m_pageActive || aiwf::AppState::Get().ShuttingDown()) break;
            JsonObject state{ nullptr };
            try { state = co_await http::GetJsonAsync(aiwf::AppState::Get().EngineApi() + L"/checkpoint-downloads/" + hstring(m_checkpointDownloadId)); }
            catch (hresult_error const&) { CheckpointDownloadStatus().Text(L"Waiting for the local engine to report download progress..."); continue; }
            auto received = static_cast<uint64_t>(http::Num(state, L"bytes_received"));
            auto total = static_cast<uint64_t>(http::Num(state, L"bytes_total"));
            std::wstring status = std::wstring(http::Str(state, L"message"));
            if (received) status += L"  " + aiwf::paths::FormatBytes(received) + (total ? L" / " + aiwf::paths::FormatBytes(total) : L"");
            CheckpointDownloadStatus().Text(hstring(status));
            auto current = std::wstring(http::Str(state, L"state"));
            if (current == L"completed" || current == L"failed" || current == L"cancelled")
            {
                m_checkpointDownloading = false; m_checkpointMonitoring = false; m_checkpointDownloadId.clear();
                CheckpointDownloadButton().IsEnabled(true); CheckpointCancelButton().IsEnabled(false);
                if (current == L"completed") RefreshBackendReadiness();
                co_return;
            }
        }
        m_checkpointMonitoring = false;
    }

    fire_and_forget SetupPage::CheckpointCancel_Click(IInspectable const&, RoutedEventArgs const&)
    {
        auto strong = get_strong();
        if (m_checkpointDownloadId.empty()) co_return;
        CheckpointCancelButton().IsEnabled(false);
        try
        {
            JsonObject empty;
            co_await http::PostJsonAsync(aiwf::AppState::Get().EngineApi() + L"/checkpoint-downloads/" + hstring(m_checkpointDownloadId) + L"/cancel", empty);
            CheckpointDownloadStatus().Text(L"Cancelling... this may take a moment while the network request returns.");
        }
        catch (hresult_error const&) { CheckpointDownloadStatus().Text(L"Could not request cancellation. Check the local engine."); }
    }

    // the person's own copy of the engine list; created from the shipped one the first time
    void SetupPage::OpenEngineList_Click(IInspectable const&, RoutedEventArgs const&)
    {
        auto userList = aiwf::paths::DataDir() / L"engines.json";
        std::error_code ignored;
        if (!fs::exists(userList, ignored))
        {
            fs::copy_file(aiwf::EngineSupervisor::Instance().ManifestPath(), userList, ignored);
        }
        auto argument = L"\"" + userList.wstring() + L"\"";
        ShellExecuteW(nullptr, L"open", L"notepad.exe", argument.c_str(), nullptr, SW_SHOWNORMAL);
    }

    fire_and_forget SetupPage::ReloadEngines_Click(IInspectable const&, RoutedEventArgs const&)
    {
        auto strong = get_strong();
        std::wstring problem;
        if (!aiwf::EngineSupervisor::Instance().Load(problem))
        {
            co_await ui::ShowMessageAsync(XamlRoot(), L"The engine list could not be read", hstring(problem));
            co_return;
        }
        LoadEngineFolders();
        ShowEngines();
        aiwf::EngineSupervisor::Instance().RefreshAsync();
    }

    // ---- step 5: review and apply ---------------------------------------------------------------------------
    std::vector<SetupPage::FolderField*> SetupPage::ChangedFields()
    {
        std::vector<FolderField*> changed;
        for (auto& field : m_fields)
        {
            if (field.available && CurrentValue(field) != field.original) changed.push_back(&field);
        }
        return changed;
    }

    void SetupPage::BuildReview()
    {
        ReviewRows().Children().Clear();
        auto changed = ChangedFields();
        auto currentBackends = aiwf::AppState::Get().EnabledImageBackends();
        auto selectedBackends = SelectedImageBackends();
        bool backendsChanged = currentBackends != selectedBackends;
        if (changed.empty() && !backendsChanged)
        {
            ReviewRows().Children().Append(ui::Text(L"No changes. Applying keeps the current folders and image backends, then marks setup as done.", L"BodyTextBlockStyle"));
            return;
        }
        if (!changed.empty()) ReviewRows().Children().Append(ui::Text(L"These folders change:", L"SubtitleTextBlockStyle"));
        // this loop lists each change as "before -> after", saying "default" for an empty value
        bool launchChanged = false;
        std::set<std::wstring> enginesChanged;
        for (auto* field : changed)
        {
            auto describe = [field](std::wstring const& value)
            {
                if (value.empty()) return std::wstring(L"default");
                return field->list ? Join(Lines(value), L';') : value;
            };
            StackPanel row;
            row.Spacing(2);
            row.Children().Append(ui::Text(hstring(field->title), L"BodyStrongTextBlockStyle"));
            row.Children().Append(ui::Text(hstring(L"Before: " + describe(field->original)), L"CaptionTextBlockStyle", 0.7));
            row.Children().Append(ui::Text(hstring(L"After: " + describe(CurrentValue(*field))), L"CaptionTextBlockStyle"));
            ReviewRows().Children().Append(row);
            if (field->engineId.empty()) launchChanged = true;
            else enginesChanged.insert(field->engineId);
        }
        if (backendsChanged)
        {
            StackPanel row;
            row.Spacing(2);
            row.Children().Append(ui::Text(L"Image backends", L"BodyStrongTextBlockStyle"));
            row.Children().Append(ui::Text(hstring(L"Before: " + DescribeImageBackends(currentBackends)), L"CaptionTextBlockStyle", 0.7));
            row.Children().Append(ui::Text(hstring(L"After: " + DescribeImageBackends(selectedBackends)), L"CaptionTextBlockStyle"));
            ReviewRows().Children().Append(row);
        }
        // what happens to running engines, so nothing restarts by surprise
        std::wstring after;
        if (launchChanged)
        {
            after += L"The engine API restarts to use the new image and model folders. If you also use AES Studio Pro in the browser, restart it to pick them up. ";
        }
        for (auto const& id : enginesChanged)
        {
            auto const* spec = aiwf::EngineSupervisor::Instance().Find(id);
            after += (spec ? spec->name : id) + L" restarts if AES Studio started it; otherwise the change applies the next time it starts from here. ";
        }
        if (backendsChanged)
        {
            after += L"Create will offer only " + DescribeImageBackends(selectedBackends) + L". This does not install backend software. ";
        }
        after += L"Files are not moved or copied.";
        ReviewRows().Children().Append(ui::Text(hstring(after), L"CaptionTextBlockStyle", 0.8));
    }

    fire_and_forget SetupPage::ApplyAsync()
    {
        if (m_applying) co_return;
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        auto changed = ChangedFields();
        if (!changed.empty() && aiwf::EngineSupervisor::Instance().IsAnswering(L"engine-api"))
        {
            JsonObject state{ nullptr }; bool checkFailed = false;
            try { state = co_await http::GetJsonAsync(aiwf::AppState::Get().EngineApi() + L"/checkpoint-downloads"); }
            catch (hresult_error const&) { checkFailed = true; }
            if (checkFailed || http::Obj(state, L"job"))
            {
                co_await ui::ShowMessageAsync(XamlRoot(), L"Wait before changing folders",
                    L"A checkpoint download may be running. Wait for it to finish or cancel it on the Engines step before applying folder changes.");
                co_return;
            }
        }

        // every new folder must be a full path; nothing is saved while one is not
        for (auto* field : changed)
        {
            auto value = CurrentValue(*field);
            for (auto const& path : field->list ? Lines(value) : std::vector<std::wstring>{ value })
            {
                if (!path.empty() && !IsFullPath(path))
                {
                    ResultBar().Severity(InfoBarSeverity::Error);
                    ResultBar().Title(L"Not saved");
                    ResultBar().Message(hstring(field->title + L": use a full folder path such as D:\\AI\\Models instead of \"" + path + L"\"."));
                    ResultBar().IsOpen(true);
                    co_return;
                }
            }
        }

        // folders that do not exist yet are created only after the person agrees
        std::vector<std::wstring> missing;
        for (auto* field : changed)
        {
            auto value = CurrentValue(*field);
            for (auto const& path : field->list ? Lines(value) : std::vector<std::wstring>{ value })
            {
                if (!path.empty() && !FolderExists(path)) missing.push_back(path);
            }
        }
        if (!missing.empty())
        {
            bool create = co_await ui::ConfirmAsync(XamlRoot(), L"Create these folders?",
                hstring(L"These folders do not exist yet:\n" + Join(missing, L'\n') + L"\n\nAES Studio can create them now."),
                L"Create folders", L"Go back");
            if (!create || aiwf::AppState::Get().ShuttingDown()) co_return;
        }

        m_applying = true;
        NextButton().IsEnabled(false);
        BackButton().IsEnabled(false);
        ResultBar().Severity(InfoBarSeverity::Informational);
        ResultBar().Title(L"Applying");
        ResultBar().Message(L"Saving folders...");
        ResultBar().IsOpen(true);

        // 1. image output and model folders -> launch.json, through the engine API
        JsonObject body;
        bool launchChanged = false;
        for (auto* field : changed)
        {
            if (!field->engineId.empty()) continue;
            launchChanged = true;
            auto value = CurrentValue(*field);
            if (field->list)
            {
                JsonArray lines;
                for (auto const& line : Lines(value)) lines.Append(JsonValue::CreateStringValue(line));
                body.Insert(hstring(field->key), lines);
            }
            else
            {
                body.Insert(hstring(field->key), JsonValue::CreateStringValue(value));
            }
        }
        hstring failure;
        if (launchChanged)
        {
            body.Insert(L"create_missing", JsonValue::CreateBooleanValue(!missing.empty()));
            try
            {
                co_await http::PostJsonAsync(aiwf::AppState::Get().EngineApi() + L"/setup/folders", body);
            }
            catch (hresult_error const& error)
            {
                failure = error.message();
            }
        }
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile
        if (!failure.empty())
        {
            m_applying = false;
            NextButton().IsEnabled(true);
            BackButton().IsEnabled(true);
            ResultBar().Severity(InfoBarSeverity::Error);
            ResultBar().Title(L"Image and model folders were not saved");
            ResultBar().Message(failure);
            ResultBar().IsOpen(true);
            co_return;
        }

        // 2. Dataset Studio and ReTrain folders -> settings.json, used when those engines start
        std::set<std::wstring> enginesChanged;
        std::vector<aiwf::AppState::EngineEnvUpdate> engineEnvUpdates;
        std::wstring createProblem;
        for (auto* field : changed)
        {
            if (field->engineId.empty()) continue;
            auto value = CurrentValue(*field);
            auto paths = field->list ? Lines(value) : std::vector<std::wstring>{ value };
            for (auto const& path : paths)
            {
                std::error_code error;
                if (!path.empty() && !FolderExists(path) && !fs::create_directories(path, error) && error)
                {
                    createProblem += path + L" (" + std::wstring(to_hstring(error.message())) + L") ";
                }
            }
            engineEnvUpdates.push_back({ field->engineId, field->key, field->list ? Join(paths, field->separator) : value });
            enginesChanged.insert(field->engineId);
            if (field->engineId == L"datasets" && (field->key == L"DATASET_STUDIO_STATE" || field->key == L"AIWF_DATASET_STUDIO_TOKEN_FILE")) enginesChanged.insert(L"engine-api");
        }

        if (!createProblem.empty())
        {
            m_applying = false;
            NextButton().IsEnabled(true);
            BackButton().IsEnabled(true);
            ResultBar().Severity(InfoBarSeverity::Error);
            ResultBar().Title(L"Some folders could not be created");
            auto message = L"AES Studio could not create: " + createProblem +
                L"Your local folder settings, backend choice, and setup status remain unchanged. ";
            if (launchChanged) message += L"The image and model folders may already have been saved by the engine API. ";
            message += L"Fix the folder permissions or choose another folder, then apply again.";
            ResultBar().Message(hstring(message));
            ResultBar().IsOpen(true);
            co_return;
        }

        // 3. commit all local settings and completion together; a failed write restores the prior state
        auto selectedBackends = SelectedImageBackends();
        if (selectedBackends.empty() || !aiwf::AppState::Get().ApplySetupSettings(selectedBackends, std::move(engineEnvUpdates)))
        {
            m_applying = false;
            NextButton().IsEnabled(true);
            BackButton().IsEnabled(true);
            ResultBar().Severity(InfoBarSeverity::Error);
            ResultBar().Title(L"Studio settings were not saved");
            ResultBar().Message(launchChanged
                ? L"AES Studio could not write %LOCALAPPDATA%\\Programs\\AES Studio Data\\settings.json, so your previous local folders, backend choice, and setup status remain active. The image and model folders may already have been saved by the engine API. Check the settings file and folder permissions, then review and apply again."
                : L"AES Studio could not write %LOCALAPPDATA%\\Programs\\AES Studio Data\\settings.json, so your previous folders, backend choice, and setup status remain active. Check the settings file and folder permissions, then review and apply again.");
            ResultBar().IsOpen(true);
            co_return;
        }

        // 4. setup is done; the window opens on Home from now on
        for (auto* field : changed) field->original = CurrentValue(*field);

        // 5. engines this app started pick the new folders up by restarting; others are left alone
        auto& supervisor = aiwf::EngineSupervisor::Instance();
        auto restart = enginesChanged;
        if (launchChanged) restart.insert(L"engine-api");
        std::wstring report = L"Your folders are saved. ";
        for (auto const& id : restart)
        {
            auto const* spec = supervisor.Find(id);
            auto name = spec ? spec->name : id;
            auto state = supervisor.Status(id).state;
            if (state == aiwf::EngineState::Running || state == aiwf::EngineState::Starting)
            {
                std::wstring ignored;
                supervisor.Stop(id, ignored);
                supervisor.StartAsync(id);
                report += name + L" is restarting with them. ";
            }
            else if (state == aiwf::EngineState::RunningExternal)
            {
                report += name + L" was started outside AES Studio; restart it there to use them. ";
            }
            else
            {
                report += name + L" uses them the next time it starts. ";
            }
        }
        if (launchChanged) report += L"AES Studio Pro, if you use it, reads them when it next starts. ";
        m_applying = false;
        m_applied = true;
        m_launchLoaded = false;   // read the saved values back the next time setup opens
        NextButton().IsEnabled(true);
        ResultBar().Severity(InfoBarSeverity::Success);
        ResultBar().Title(L"Setup complete");
        ResultBar().Message(hstring(report));
        HomeButton().Visibility(Visibility::Visible);
        ShowStep(m_step);
    }
}
