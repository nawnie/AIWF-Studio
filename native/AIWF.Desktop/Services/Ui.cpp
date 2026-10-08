// Shared UI helpers (see Ui.h).
#include "pch.h"
#include <shellapi.h>
#include <shlobj.h>
#include "Services/Ui.h"
#include "Services/AppState.h"
#include "Services/EngineSupervisor.h"
#include "Services/Http.h"

using namespace winrt;
using namespace winrt::Windows::Foundation;
using namespace winrt::Microsoft::UI::Xaml;
using namespace winrt::Microsoft::UI::Xaml::Controls;
using namespace winrt::Microsoft::UI::Xaml::Media;

namespace aiwf::ui
{
    // ---- building blocks ----------------------------------------------------------------------------
    SolidColorBrush ColorBrush(uint32_t argb)
    {
        return SolidColorBrush(winrt::Windows::UI::Color{ static_cast<uint8_t>(argb >> 24), static_cast<uint8_t>(argb >> 16),
                                                          static_cast<uint8_t>(argb >> 8), static_cast<uint8_t>(argb) });
    }

    Style AppStyle(wchar_t const* key)
    {
        return Application::Current().Resources().Lookup(box_value(key)).as<Style>();
    }

    TextBlock Text(hstring const& text, wchar_t const* style, double opacity)
    {
        TextBlock block;
        if (style) block.Style(AppStyle(style));
        block.Text(text);
        block.TextWrapping(TextWrapping::Wrap);
        block.Opacity(opacity);
        return block;
    }

    Border Card(UIElement const& content)
    {
        Border border;
        border.Style(AppStyle(L"CardStyle"));
        border.Child(content);
        return border;
    }

    Border Chip(hstring const& text, uint32_t dotArgb)
    {
        StackPanel row;
        row.Orientation(Orientation::Horizontal);
        row.Spacing(6);
        if (dotArgb != 0)
        {
            Shapes::Ellipse dot;
            dot.Width(8);
            dot.Height(8);
            dot.Fill(ColorBrush(dotArgb));
            dot.VerticalAlignment(VerticalAlignment::Center);
            row.Children().Append(dot);
        }
        auto label = Text(text, L"CaptionTextBlockStyle");
        label.TextWrapping(TextWrapping::NoWrap);
        row.Children().Append(label);
        Border chip;
        chip.Style(AppStyle(L"ChipStyle"));
        chip.Child(row);
        return chip;
    }

    FontIcon Icon(wchar_t const* glyph, double size)
    {
        FontIcon icon;
        icon.Glyph(glyph);
        icon.FontSize(size);
        return icon;
    }

    Button IconButton(wchar_t const* glyph, hstring const& label, bool accent)
    {
        StackPanel content;
        content.Orientation(Orientation::Horizontal);
        content.Spacing(8);
        content.Children().Append(Icon(glyph));
        TextBlock text;
        text.Text(label);
        content.Children().Append(text);
        Button button;
        if (accent) button.Style(AppStyle(L"AccentButtonStyle"));
        button.Content(content);
        // screen readers announce the label, not the icon glyph
        Automation::AutomationProperties::SetName(button, label);
        return button;
    }

    hstring SelectedTag(ComboBox const& box)
    {
        auto item = box.SelectedItem().try_as<ComboBoxItem>();
        return item ? unbox_value_or<hstring>(item.Tag(), L"") : hstring();
    }

    // ---- messages ------------------------------------------------------------------------------------
    IAsyncAction ShowMessageAsync(XamlRoot root, hstring title, hstring message)
    {
        ContentDialog dialog;
        dialog.XamlRoot(root);
        dialog.Title(box_value(title));
        TextBlock body;
        body.Text(message);
        body.TextWrapping(TextWrapping::Wrap);
        body.IsTextSelectionEnabled(true);
        dialog.Content(body);
        dialog.CloseButtonText(L"OK");
        dialog.DefaultButton(ContentDialogButton::Close);
        try
        {
            co_await dialog.ShowAsync();
        }
        catch (hresult_error const&)
        {
            // only one dialog can be open at a time; a second message waits for the next action
        }
    }

    IAsyncOperation<bool> ConfirmAsync(XamlRoot root, hstring title, hstring message, hstring primary, hstring close)
    {
        ContentDialog dialog;
        dialog.XamlRoot(root);
        dialog.Title(box_value(title));
        TextBlock body;
        body.Text(message);
        body.TextWrapping(TextWrapping::Wrap);
        dialog.Content(body);
        dialog.PrimaryButtonText(primary);
        dialog.CloseButtonText(close);
        dialog.DefaultButton(ContentDialogButton::Close);   // the safe choice is the default
        try
        {
            co_return co_await dialog.ShowAsync() == ContentDialogResult::Primary;
        }
        catch (hresult_error const&)
        {
            co_return false;
        }
    }

    // ---- choosing a folder ---------------------------------------------------------------------------------
    IAsyncOperation<hstring> PickFolderAsync(XamlRoot root)
    {
        winrt::Windows::Storage::Pickers::FolderPicker picker;
        picker.SuggestedStartLocation(winrt::Windows::Storage::Pickers::PickerLocationId::ComputerFolder);
        picker.FileTypeFilter().Append(L"*");
        // an unpackaged app must tell the picker which window owns it, or the picker refuses to open
        HWND owner = nullptr;
        if (root && root.ContentIslandEnvironment())
        {
            owner = winrt::Microsoft::UI::GetWindowFromWindowId(root.ContentIslandEnvironment().AppWindowId());
        }
        if (!owner) owner = GetActiveWindow();
        picker.as<::IInitializeWithWindow>()->Initialize(owner);
        try
        {
            auto folder = co_await picker.PickSingleFolderAsync();
            co_return folder ? folder.Path() : hstring{};
        }
        catch (hresult_error const&)
        {
            co_return hstring{};
        }
    }

    // ---- the Hugging Face key ---------------------------------------------------------------------------
    IAsyncOperation<bool> AskForHfKeyAsync(XamlRoot root, hstring reason, bool hasKey)
    {
        using namespace winrt::Windows::Data::Json;
        auto api = AppState::Get().EngineApi();
        ContentDialog dialog;
        dialog.XamlRoot(root);
        dialog.Title(box_value(L"Hugging Face key"));
        StackPanel content;
        content.Spacing(10);
        content.Children().Append(Text(reason));
        content.Children().Append(Text(L"AIWF Studio never displays, logs or sends the key anywhere except to Hugging Face for this check.",
                                       L"CaptionTextBlockStyle", 0.7));
        PasswordBox box;
        box.PlaceholderText(L"hf_...");
        Automation::AutomationProperties::SetName(box, L"Hugging Face key");
        content.Children().Append(box);
        auto error = Text(L"", L"CaptionTextBlockStyle");
        error.Foreground(ColorBrush(0xFFD13438));
        content.Children().Append(error);
        dialog.Content(content);
        dialog.PrimaryButtonText(L"Save key");
        if (hasKey) dialog.SecondaryButtonText(L"Remove saved key");
        dialog.CloseButtonText(L"Cancel");
        dialog.DefaultButton(ContentDialogButton::Primary);

        // the dialog stays open while the key is checked, and shows the engine's answer if it is refused
        auto saved = std::make_shared<bool>(false);
        dialog.PrimaryButtonClick([box, error, api, saved](ContentDialog, ContentDialogButtonClickEventArgs args) -> fire_and_forget
        {
            auto deferral = args.GetDeferral();
            auto token = box.Password();
            if (token.empty())
            {
                error.Text(L"Paste a key first.");
                args.Cancel(true);
                deferral.Complete();
                co_return;
            }
            error.Text(L"Checking with Hugging Face...");
            JsonObject body;
            body.Insert(L"token", JsonValue::CreateStringValue(token));
            hstring failure;
            try
            {
                co_await http::PostJsonAsync(api + L"/models/hf-token", body);
            }
            catch (hresult_error const& problem)
            {
                failure = problem.message();
            }
            if (failure.empty())
            {
                *saved = true;
                box.Password(L"");
            }
            else
            {
                error.Text(failure);
                args.Cancel(true);
            }
            deferral.Complete();
        });
        auto result = ContentDialogResult::None;
        try
        {
            result = co_await dialog.ShowAsync();
        }
        catch (hresult_error const&)
        {
            co_return false;   // another dialog was already open
        }
        if (result == ContentDialogResult::Secondary)
        {
            try
            {
                co_await http::PostJsonAsync(api + L"/models/hf-token/clear", JsonObject());
            }
            catch (hresult_error const&)
            {
            }
            co_return true;
        }
        co_return *saved;
    }

    // ---- engines a page depends on -----------------------------------------------------------------
    bool UpdateEngineBar(InfoBar const& bar, std::vector<std::wstring> const& engineIds)
    {
        auto& supervisor = EngineSupervisor::Instance();
        std::vector<std::wstring> off;
        bool busy = false;
        std::wstring failure;
        // this loop collects the engines that are not answering and why
        for (auto const& id : engineIds)
        {
            auto status = supervisor.Status(id);
            if (status.state == EngineState::Running || status.state == EngineState::RunningExternal) continue;
            auto spec = supervisor.Find(id);
            auto name = spec ? spec->name : id;
            off.push_back(name);
            if (status.state == EngineState::Starting || status.state == EngineState::Unknown) busy = true;
            if (status.state == EngineState::Failed && failure.empty()) failure = name + L": " + status.detail;
        }
        if (off.empty())
        {
            bar.IsOpen(false);
            return true;
        }
        std::wstring names;
        for (size_t i = 0; i < off.size(); ++i)
        {
            if (i > 0) names += (i + 1 == off.size()) ? L" and " : L", ";
            names += off[i];
        }
        if (busy)
        {
            bar.Title(L"Starting");
            bar.Message(names + L" " + (off.size() > 1 ? L"are" : L"is") + L" starting or being checked...");
            bar.Severity(InfoBarSeverity::Informational);
        }
        else if (!failure.empty())
        {
            bar.Title(L"An engine has a problem");
            bar.Message(failure);
            bar.Severity(InfoBarSeverity::Warning);
        }
        else
        {
            bar.Title(L"Engine off");
            bar.Message(names + L" " + (off.size() > 1 ? L"are" : L"is") + L" not running. Start " + (off.size() > 1 ? L"them" : L"it") + L" to use this page.");
            bar.Severity(InfoBarSeverity::Informational);
        }
        if (auto button = bar.ActionButton().try_as<Button>()) button.IsEnabled(!busy);
        bar.IsOpen(true);
        return false;
    }

    IAsyncAction StartEnginesAsync(std::vector<std::wstring> engineIds)
    {
        auto generation = EngineSupervisor::Instance().StopGeneration();
        // the engine API goes first: the other pages reach their engines through it
        std::stable_sort(engineIds.begin(), engineIds.end(),
                         [](std::wstring const& a, std::wstring const& b) { return a == L"engine-api" && b != L"engine-api"; });
        for (auto const& id : engineIds)
        {
            if (AppState::Get().ShuttingDown() || generation != EngineSupervisor::Instance().StopGeneration()) co_return;
            co_await EngineSupervisor::Instance().StartAsync(id, generation);
        }
    }

    // ---- shell -------------------------------------------------------------------------------------------
    void OpenWithShell(std::wstring const& path)
    {
        ShellExecuteW(nullptr, L"open", path.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    }

    void ShowInExplorer(std::wstring const& path)
    {
        // reuses an open Explorer window on that folder when there is one
        if (PIDLIST_ABSOLUTE item = ILCreateFromPathW(path.c_str()))
        {
            SHOpenFolderAndSelectItems(item, 0, nullptr, 0);
            ILFree(item);
        }
    }

    hstring FriendlyTime(hstring const& iso)
    {
        int year = 0, month = 0, day = 0, hour = 0, minute = 0, second = 0;
        if (swscanf_s(iso.c_str(), L"%d-%d-%dT%d:%d:%d", &year, &month, &day, &hour, &minute, &second) != 6) return iso;
        SYSTEMTIME utc{ static_cast<WORD>(year), static_cast<WORD>(month), 0, static_cast<WORD>(day),
                        static_cast<WORD>(hour), static_cast<WORD>(minute), static_cast<WORD>(second), 0 };
        SYSTEMTIME local{};
        if (!SystemTimeToTzSpecificLocalTime(nullptr, &utc, &local)) return iso;
        wchar_t date[64]{};
        wchar_t time[64]{};
        GetDateFormatEx(LOCALE_NAME_USER_DEFAULT, 0, &local, L"MMM d", date, 64, nullptr);
        GetTimeFormatEx(LOCALE_NAME_USER_DEFAULT, TIME_NOSECONDS, &local, nullptr, time, 64);
        return hstring(date) + L", " + time;
    }
}
