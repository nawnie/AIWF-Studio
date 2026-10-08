// Datasets page implementation.
#include "pch.h"
#include "DatasetsPage.xaml.h"
#if __has_include("DatasetsPage.g.cpp")
#include "DatasetsPage.g.cpp"
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
        // "12 train · 3 validation rows" from a package's counts
        std::wstring RowsText(JsonObject const& counts)
        {
            auto train = static_cast<int>(http::Num(counts, L"train"));
            auto validation = static_cast<int>(http::Num(counts, L"validation"));
            return std::to_wstring(train) + L" train · " + std::to_wstring(validation) + L" validation rows";
        }

        fire_and_forget LoadThumbnailAsync(BitmapImage bitmap, std::wstring path, Microsoft::UI::Dispatching::DispatcherQueue dispatcher)
        {
            try
            {
                auto file = co_await StorageFile::GetFileFromPathAsync(path);
                auto stream = co_await file.OpenReadAsync();
                co_await wil::resume_foreground(dispatcher);
                if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile
                co_await bitmap.SetSourceAsync(stream);
            }
            catch (hresult_error const&)
            {
                // a file that moved shows an empty tile
            }
        }
    }

    DatasetsPage::DatasetsPage()
    {
        InitializeComponent();
    }

    // ---- page lifetime ------------------------------------------------------------------------------------
    void DatasetsPage::OnNavigatedTo(NavigationEventArgs const&)
    {
        auto weak = get_weak();
        auto dispatcher = DispatcherQueue();
        m_statusToken = aiwf::EngineSupervisor::Instance().StatusChanged([weak, dispatcher]
        {
            dispatcher.TryEnqueue([weak]
            {
                if (auto self = weak.get(); self && !aiwf::AppState::Get().ShuttingDown()) self->UpdateEngineState();
            });
        });
        m_loaded = false;
        UpdateEngineState();
    }

    void DatasetsPage::OnNavigatedFrom(NavigationEventArgs const&)
    {
        aiwf::EngineSupervisor::Instance().StatusChanged(m_statusToken);
    }

    void DatasetsPage::UpdateEngineState()
    {
        bool ready = ui::UpdateEngineBar(EngineBar(), { L"engine-api", L"datasets" });
        if (ready && !m_loaded) LoadAsync();
        // an empty page while the engines are off should read as "unavailable", not "you have nothing"
        if (!ready && !m_loaded && !m_loading)
        {
            PackagesNote().Text(L"Packages and recent images are listed once the engine API and the dataset engine are running (see above).");
        }
        UpdateCatalogButton();
    }

    void DatasetsPage::StartEngine_Click(IInspectable const&, RoutedEventArgs const&)
    {
        ui::StartEnginesAsync({ L"engine-api", L"datasets" });
    }

    void DatasetsPage::Refresh_Click(IInspectable const&, RoutedEventArgs const&)
    {
        LoadAsync();
    }

    // ---- loading packages and recent images --------------------------------------------------------------------
    fire_and_forget DatasetsPage::LoadAsync()
    {
        if (m_loading) co_return;
        m_loading = true;
        m_loaded = true;
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        auto& state = aiwf::AppState::Get();
        JsonObject packages{ nullptr };
        JsonObject outputs{ nullptr };
        JsonObject root{ nullptr };
        hstring failure;
        try
        {
            packages = co_await http::GetJsonAsync(state.EngineApi() + L"/datasets/packages");
            root = co_await http::GetJsonAsync(state.EngineApiRoot() + L"/");
            outputs = co_await http::GetJsonAsync(state.EngineApi() + L"/outputs?limit=40");
        }
        catch (hresult_error const& error)
        {
            failure = error.message();
        }
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile
        m_loading = false;
        Packages().Children().Clear();
        if (!failure.empty())
        {
            m_loaded = false;
            PackagesNote().Text(L"Could not load datasets: " + failure);
            co_return;
        }

        // this loop builds one card per package; broken ones say why instead of offering training
        int ready = 0;
        auto weak = get_weak();
        for (auto const& value : http::Arr(packages, L"packages"))
        {
            auto package = value.GetObject();
            auto name = http::Str(package, L"package_name");
            bool isReady = http::Str(package, L"status") == L"ready";
            Grid grid;
            grid.ColumnSpacing(12);
            ColumnDefinition textColumn;
            textColumn.Width(GridLengthHelper::FromValueAndType(1, GridUnitType::Star));
            ColumnDefinition buttonColumn;
            buttonColumn.Width(GridLengthHelper::Auto());
            grid.ColumnDefinitions().Append(textColumn);
            grid.ColumnDefinitions().Append(buttonColumn);

            StackPanel text;
            text.Spacing(6);
            text.Children().Append(ui::Text(name, L"BodyStrongTextBlockStyle"));
            if (isReady)
            {
                ++ready;
                auto sha = std::wstring(http::Str(package, L"manifest_sha256"));
                StackPanel chips;
                chips.Orientation(Orientation::Horizontal);
                chips.Spacing(8);
                chips.Children().Append(ui::Chip(hstring(L"revision " + sha.substr(0, 12))));
                chips.Children().Append(ui::Chip(hstring(RowsText(http::Obj(package, L"counts")))));
                chips.Children().Append(ui::Chip(http::Str(package, L"modality", L"text-only")));
                auto created = http::Str(package, L"created_at");
                if (!created.empty()) chips.Children().Append(ui::Text(L"published " + ui::FriendlyTime(created), L"CaptionTextBlockStyle", 0.6));
                text.Children().Append(chips);

                auto use = ui::IconButton(L"", L"Use for training");
                use.VerticalAlignment(VerticalAlignment::Center);
                use.Click([weak, packageName = std::wstring(name), sha](auto&&, auto&&)
                {
                    // the exact revision travels with the name, so Train can never pick a newer one by accident
                    aiwf::AppState::Get().SetPendingPackage({ packageName, sha });
                    aiwf::AppState::Get().RequestNavigation(L"train");
                });
                Grid::SetColumn(use, 1);
                grid.Children().Append(use);
            }
            else
            {
                text.Children().Append(ui::Text(L"Not usable: " + http::Str(package, L"reason"), L"CaptionTextBlockStyle", 0.75));
            }
            grid.Children().Append(text);
            Packages().Children().Append(ui::Card(grid));
        }
        PackagesNote().Text(ready > 0 ? L"" : L"No published packages yet. Packages are built in Dataset Studio's Export view from a collection of text examples.");

        // this loop fills the image picker from Studio's recent outputs
        Outputs().Items().Clear();
        auto outputDir = std::wstring(http::Str(root, L"output_dir"));
        for (auto const& value : http::Arr(outputs, L"outputs"))
        {
            auto output = value.GetObject();
            auto relative = std::wstring(http::Str(output, L"relative_path"));
            auto path = relative;
            std::replace(path.begin(), path.end(), L'/', L'\\');
            path = (std::filesystem::path(outputDir) / path).wstring();

            Grid tile;
            tile.Width(176);
            tile.Height(150);
            tile.Tag(box_value(hstring(relative)));
            RowDefinition imageRow;
            imageRow.Height(GridLengthHelper::FromValueAndType(1, GridUnitType::Star));
            RowDefinition captionRow;
            captionRow.Height(GridLengthHelper::Auto());
            tile.RowDefinitions().Append(imageRow);
            tile.RowDefinitions().Append(captionRow);
            Image image;
            image.Stretch(Media::Stretch::UniformToFill);
            BitmapImage bitmap;
            bitmap.DecodePixelHeight(220);
            image.Source(bitmap);
            LoadThumbnailAsync(bitmap, path, dispatcher);
            tile.Children().Append(image);
            auto prompt = http::Str(output, L"prompt");
            auto caption = ui::Text(prompt.empty() ? hstring(relative) : prompt, L"CaptionTextBlockStyle", 0.75);
            caption.TextWrapping(TextWrapping::NoWrap);
            caption.TextTrimming(TextTrimming::CharacterEllipsis);
            caption.Margin(ThicknessHelper::FromLengths(4, 4, 4, 2));
            Grid::SetRow(caption, 1);
            tile.Children().Append(caption);
            ToolTipService::SetToolTip(tile, box_value(prompt.empty() ? hstring(relative) : prompt));
            Automation::AutomationProperties::SetName(tile, prompt.empty() ? hstring(relative) : prompt);
            Outputs().Items().Append(tile);
        }
        UpdateCatalogButton();
    }

    // ---- cataloging --------------------------------------------------------------------------------------------
    void DatasetsPage::Outputs_SelectionChanged(IInspectable const&, SelectionChangedEventArgs const&)
    {
        UpdateCatalogButton();
    }

    void DatasetsPage::UpdateCatalogButton()
    {
        auto count = Outputs().SelectedItems().Size();
        CatalogButtonText().Text(count > 0 ? hstring(L"Catalog " + std::to_wstring(count) + (count == 1 ? L" image" : L" images")) : hstring(L"Catalog selected"));
        CatalogButton().IsEnabled(count > 0 && aiwf::EngineSupervisor::Instance().IsAnswering(L"datasets"));
    }

    fire_and_forget DatasetsPage::Catalog_Click(IInspectable const&, RoutedEventArgs const&)
    {
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        auto& state = aiwf::AppState::Get();
        auto projectId = state.ProjectId();
        if (projectId.empty())
        {
            CatalogResult().Severity(InfoBarSeverity::Warning);
            CatalogResult().Title(L"Choose a project first");
            CatalogResult().Message(L"Cataloged images are tagged with the project in focus. Pick one in the title bar or create one on the Projects page.");
            CatalogResult().IsOpen(true);
            co_return;
        }
        JsonArray paths;
        for (auto const& item : Outputs().SelectedItems())
        {
            if (auto tile = item.try_as<FrameworkElement>()) paths.Append(JsonValue::CreateStringValue(unbox_value_or<hstring>(tile.Tag(), L"")));
        }
        JsonObject body;
        body.Insert(L"output_paths", paths);
        CatalogButton().IsEnabled(false);

        JsonObject result{ nullptr };
        hstring failure;
        try
        {
            result = co_await http::PostJsonAsync(state.EngineApi() + L"/projects/" + http::EscapeSegment(hstring(projectId)) + L"/catalog-outputs", body);
        }
        catch (hresult_error const& error)
        {
            failure = error.message();
        }
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile
        if (!failure.empty())
        {
            CatalogResult().Severity(InfoBarSeverity::Error);
            CatalogResult().Title(L"Not cataloged");
            CatalogResult().Message(failure);
        }
        else
        {
            auto assets = http::Arr(result, L"assets");
            auto collection = http::Obj(result, L"collection");
            CatalogResult().Severity(InfoBarSeverity::Success);
            CatalogResult().Title(L"Cataloged");
            CatalogResult().Message(hstring(std::to_wstring(assets.Size()) + L" image(s) are in the dataset engine's collection “") +
                                    http::Str(collection, L"name") + L"”, tagged " + http::Str(result, L"project_tag") +
                                    L". Images are cataloged only; training packages stay text-only.");
            Outputs().SelectedItems().Clear();
        }
        CatalogResult().IsOpen(true);
        UpdateCatalogButton();
    }
}
