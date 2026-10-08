// Projects page implementation.
#include "pch.h"
#include "ProjectsPage.xaml.h"
#if __has_include("ProjectsPage.g.cpp")
#include "ProjectsPage.g.cpp"
#endif
#include "Services/AppState.h"
#include "Services/EngineSupervisor.h"
#include "Services/Http.h"
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
        // one sentence per recorded event, in the words a person would use
        std::wstring Describe(JsonObject const& event)
        {
            auto kind = std::wstring(http::Str(event, L"kind"));
            if (kind == L"image_generated")
            {
                auto count = http::Arr(event, L"images").Size();
                return L"Generated " + std::to_wstring(count) + (count == 1 ? L" image" : L" images") + L" with Qwen Image 2.1 (seed " +
                       std::wstring(http::Str(event, L"seed")) + L")";
            }
            if (kind == L"dataset_catalog")
            {
                auto count = http::Arr(event, L"outputs").Size();
                return L"Cataloged " + std::to_wstring(count) + (count == 1 ? L" image" : L" images") + L" in collection “" +
                       std::wstring(http::Str(http::Obj(event, L"collection"), L"name")) + L"”";
            }
            if (kind == L"retrain_import")
            {
                auto dataset = http::Obj(event, L"dataset");
                auto sha = std::wstring(http::Str(dataset, L"manifest_sha256"));
                return L"Imported package “" + std::wstring(http::Str(dataset, L"package_name")) + L"” revision " + sha.substr(0, (std::min<size_t>)(12, sha.size())) +
                       L" for training";
            }
            if (kind == L"retrain_preflight")
            {
                return L"Checked a training plan with " + std::wstring(http::Str(event, L"model_id")) + L": " + std::wstring(http::Str(event, L"plan_status"));
            }
            if (kind == L"qwen_context_sent")
            {
                // events recorded before questions were stored in the ledger carry no question text
                auto question = std::wstring(http::Str(event, L"question"));
                auto model = std::wstring(http::Str(event, L"model_id"));
                if (question.empty()) return L"Asked " + model + L" a question with this project's context";
                return L"Asked " + model + L" with project context: “" + question + L"”";
            }
            return kind;
        }
    }

    ProjectsPage::ProjectsPage()
    {
        InitializeComponent();
        // re-arrange list and details whenever the page width crosses the stacking threshold
        RootGrid().SizeChanged([weak = get_weak()](auto&&, SizeChangedEventArgs const& args)
        {
            if (auto self = weak.get(); self && !aiwf::AppState::Get().ShuttingDown()) self->ApplyLayout(args.NewSize().Width);
        });
    }

    // ---- layout: list beside details when wide, list above details when narrow ------------------------------------
    void ProjectsPage::ApplyLayout(double width)
    {
        bool stacked = width < StackBelowWidth;
        if (m_layoutKnown && stacked == m_stacked) return;
        m_layoutKnown = true;
        m_stacked = stacked;
        ListColumn().Width(stacked ? GridLengthHelper::FromValueAndType(1, GridUnitType::Star) : GridLengthHelper::FromPixels(380));
        DetailColumn().Width(stacked ? GridLengthHelper::FromPixels(0) : GridLengthHelper::FromValueAndType(1, GridUnitType::Star));
        // stacked, the list keeps a bounded height (it scrolls inside) and the details take the rest
        ListRow().Height(stacked ? GridLengthHelper::Auto() : GridLengthHelper::FromValueAndType(1, GridUnitType::Star));
        DetailRow().Height(stacked ? GridLengthHelper::FromValueAndType(1, GridUnitType::Star) : GridLengthHelper::FromPixels(0));
        ListPane().MaxHeight(stacked ? 300 : std::numeric_limits<double>::infinity());
        RootGrid().ColumnSpacing(stacked ? 0 : 28);
        Grid::SetRow(DetailPane(), stacked ? 2 : 1);
        Grid::SetColumn(DetailPane(), stacked ? 0 : 1);
    }

    // ---- page lifetime ------------------------------------------------------------------------------------
    void ProjectsPage::OnNavigatedTo(NavigationEventArgs const&)
    {
        auto weak = get_weak();
        auto dispatcher = DispatcherQueue();
        m_projectToken = aiwf::AppState::Get().ProjectChanged([weak]
        {
            if (auto self = weak.get(); self && !aiwf::AppState::Get().ShuttingDown()) self->SyncProject();
        });
        m_statusToken = aiwf::EngineSupervisor::Instance().StatusChanged([weak, dispatcher]
        {
            dispatcher.TryEnqueue([weak]
            {
                if (auto self = weak.get(); self && !aiwf::AppState::Get().ShuttingDown()) self->UpdateEngineState();
            });
        });
        m_loaded = false;
        SyncProject();
        UpdateEngineState();
    }

    void ProjectsPage::OnNavigatedFrom(NavigationEventArgs const&)
    {
        aiwf::EngineSupervisor::Instance().StatusChanged(m_statusToken);
        aiwf::AppState::Get().ProjectChanged(m_projectToken);
        ++m_detailGeneration;
    }

    void ProjectsPage::SyncProject()
    {
        auto& state = aiwf::AppState::Get();
        auto id = state.ProjectId();
        m_suppressSelection = true;
        ProjectList().SelectedItem(nullptr);
        for (auto const& item : ProjectList().Items())
        {
            auto row = item.try_as<FrameworkElement>();
            if (!row) continue;
            auto tag = std::wstring(unbox_value_or<hstring>(row.Tag(), L""));
            if (tag.substr(0, tag.find(L'|')) == id) ProjectList().SelectedItem(row);
        }
        m_suppressSelection = false;
        if (!id.empty()) ShowDetailAsync(id, state.ProjectName());
        else
        {
            ++m_detailGeneration;
            DetailName().Text(L"");
            DetailId().Text(L"");
            Events().Children().Clear();
        }
    }

    void ProjectsPage::UpdateEngineState()
    {
        bool ready = ui::UpdateEngineBar(EngineBar(), { L"engine-api" });
        if (ready && !m_loaded) LoadAsync();
    }

    void ProjectsPage::StartEngine_Click(IInspectable const&, RoutedEventArgs const&)
    {
        ui::StartEnginesAsync({ L"engine-api" });
    }

    // ---- the list ------------------------------------------------------------------------------------------------
    fire_and_forget ProjectsPage::LoadAsync()
    {
        m_loaded = true;
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        JsonObject body{ nullptr };
        hstring failure;
        try
        {
            body = co_await http::GetJsonAsync(aiwf::AppState::Get().EngineApi() + L"/projects");
        }
        catch (hresult_error const& error)
        {
            failure = error.message();
        }
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile
        if (!failure.empty())
        {
            m_loaded = false;
            DetailName().Text(L"Projects are not available: " + failure);
            co_return;
        }
        auto current = aiwf::AppState::Get().ProjectId();
        m_suppressSelection = true;
        ProjectList().Items().Clear();
        // this loop lists projects with how much each has recorded
        for (auto const& value : http::Arr(body, L"projects"))
        {
            auto project = value.GetObject();
            auto id = http::Str(project, L"project_id");
            auto name = http::Str(project, L"name");
            int total = 0;
            for (auto const& pair : http::Obj(project, L"event_counts")) total += static_cast<int>(pair.Value().GetNumber());
            StackPanel row;
            row.Padding(ThicknessHelper::FromLengths(0, 6, 0, 6));
            row.Children().Append(ui::Text(name, L"BodyStrongTextBlockStyle"));
            row.Children().Append(ui::Text(L"Created " + ui::FriendlyTime(http::Str(project, L"created_at")) + L"  ·  " +
                                               to_hstring(total) + (total == 1 ? L" recorded action" : L" recorded actions"),
                                           L"CaptionTextBlockStyle", 0.7));
            row.Tag(box_value(id + L"|" + name));
            Automation::AutomationProperties::SetName(row, name);
            ProjectList().Items().Append(row);
            if (id == current) ProjectList().SelectedItem(row);
        }
        m_suppressSelection = false;
        if (!current.empty()) ShowDetailAsync(current, aiwf::AppState::Get().ProjectName());
    }

    void ProjectsPage::ProjectList_SelectionChanged(IInspectable const&, SelectionChangedEventArgs const&)
    {
        if (m_suppressSelection) return;
        auto row = ProjectList().SelectedItem().try_as<FrameworkElement>();
        if (!row) return;
        auto tag = std::wstring(unbox_value_or<hstring>(row.Tag(), L""));
        auto bar = tag.find(L'|');
        if (bar == std::wstring::npos) return;
        auto id = tag.substr(0, bar);
        auto name = tag.substr(bar + 1);
        // choosing a project here puts it in focus everywhere (the title-bar picker follows)
        aiwf::AppState::Get().SetProject(id, name);
    }

    // ---- creating -----------------------------------------------------------------------------------------------
    void ProjectsPage::Create_Click(IInspectable const&, RoutedEventArgs const&)
    {
        CreateAsync();
    }

    void ProjectsPage::NewName_KeyDown(IInspectable const&, Microsoft::UI::Xaml::Input::KeyRoutedEventArgs const& args)
    {
        if (args.Key() == Windows::System::VirtualKey::Enter)
        {
            args.Handled(true);
            CreateAsync();
        }
    }

    fire_and_forget ProjectsPage::CreateAsync()
    {
        std::wstring name{ NewName().Text() };
        while (!name.empty() && iswspace(name.back())) name.pop_back();
        while (!name.empty() && iswspace(name.front())) name.erase(0, 1);
        if (name.empty()) co_return;
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        JsonObject body;
        body.Insert(L"name", JsonValue::CreateStringValue(name));
        JsonObject created{ nullptr };
        hstring failure;
        try
        {
            created = co_await http::PostJsonAsync(aiwf::AppState::Get().EngineApi() + L"/projects", body);
        }
        catch (hresult_error const& error)
        {
            failure = error.message();
        }
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile
        if (!failure.empty())
        {
            co_await ui::ShowMessageAsync(XamlRoot(), L"Project not created", failure);
            co_return;
        }
        NewName().Text(L"");
        auto project = http::Obj(created, L"project");
        // the new project goes straight into focus
        aiwf::AppState::Get().SetProject(std::wstring(http::Str(project, L"project_id")), std::wstring(http::Str(project, L"name")));
        LoadAsync();
    }

    // ---- the activity timeline ----------------------------------------------------------------------------------
    fire_and_forget ProjectsPage::ShowDetailAsync(std::wstring id, std::wstring name)
    {
        auto generation = ++m_detailGeneration;
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        DetailName().Text(name);
        DetailId().Text(L"Project ID " + hstring(id) + L"  (in focus)");
        Events().Children().Clear();
        JsonObject body{ nullptr };
        hstring failure;
        try
        {
            body = co_await http::GetJsonAsync(aiwf::AppState::Get().EngineApi() + L"/projects/" + http::EscapeSegment(hstring(id)));
        }
        catch (hresult_error const& error)
        {
            failure = error.message();
        }
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile
        if (generation != m_detailGeneration || id != aiwf::AppState::Get().ProjectId()) co_return;
        if (!failure.empty())
        {
            Events().Children().Append(ui::Text(L"Activity is not available: " + failure, L"CaptionTextBlockStyle", 0.7));
            co_return;
        }
        auto events = http::Arr(http::Obj(body, L"project"), L"events");
        if (events.Size() == 0)
        {
            Events().Children().Append(ui::Text(L"Nothing recorded yet. Create an image, catalog outputs, import a package or check a plan.",
                                                L"BodyTextBlockStyle", 0.7));
            co_return;
        }
        // newest first
        for (int32_t i = static_cast<int32_t>(events.Size()) - 1; i >= 0; --i)
        {
            auto event = events.GetObjectAt(static_cast<uint32_t>(i));
            StackPanel row;
            row.Children().Append(ui::Text(hstring(Describe(event))));
            row.Children().Append(ui::Text(ui::FriendlyTime(http::Str(event, L"at")), L"CaptionTextBlockStyle", 0.6));
            Events().Children().Append(row);
        }
    }
}
