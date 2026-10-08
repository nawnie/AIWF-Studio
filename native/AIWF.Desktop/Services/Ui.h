// Small UI helpers shared by the pages: themed text and cards built in code, chips, dialogs,
// and the "this page needs an engine that is off" bar with its Start button.
#pragma once

namespace aiwf::ui
{
    // ---- building blocks for content created in code -------------------------------------------
    winrt::Microsoft::UI::Xaml::Media::SolidColorBrush ColorBrush(uint32_t argb);
    winrt::Microsoft::UI::Xaml::Style AppStyle(wchar_t const* key);
    winrt::Microsoft::UI::Xaml::Controls::TextBlock Text(winrt::hstring const& text,
                                                         wchar_t const* style = L"BodyTextBlockStyle",
                                                         double opacity = 1.0);
    winrt::Microsoft::UI::Xaml::Controls::Border Card(winrt::Microsoft::UI::Xaml::UIElement const& content);
    // a rounded label; dotArgb != 0 adds a colored dot in front of the text
    winrt::Microsoft::UI::Xaml::Controls::Border Chip(winrt::hstring const& text, uint32_t dotArgb = 0);
    winrt::Microsoft::UI::Xaml::Controls::FontIcon Icon(wchar_t const* glyph, double size = 14);
    // a button showing an icon and a label side by side
    winrt::Microsoft::UI::Xaml::Controls::Button IconButton(wchar_t const* glyph, winrt::hstring const& label, bool accent = false);
    // the Tag (an id) of a combo box's selected ComboBoxItem, or empty
    winrt::hstring SelectedTag(winrt::Microsoft::UI::Xaml::Controls::ComboBox const& box);

    // ---- messages ----------------------------------------------------------------------------------
    winrt::Windows::Foundation::IAsyncAction ShowMessageAsync(winrt::Microsoft::UI::Xaml::XamlRoot root,
                                                              winrt::hstring title, winrt::hstring message);
    // asks a yes/no question; true when the person chose the primary action
    winrt::Windows::Foundation::IAsyncOperation<bool> ConfirmAsync(winrt::Microsoft::UI::Xaml::XamlRoot root, winrt::hstring title,
                                                                   winrt::hstring message, winrt::hstring primary,
                                                                   winrt::hstring close = L"Not now");

    // the system folder picker, owned by this app's window; empty when the person cancels
    winrt::Windows::Foundation::IAsyncOperation<winrt::hstring> PickFolderAsync(winrt::Microsoft::UI::Xaml::XamlRoot root);

    // The Hugging Face key dialog shared by Train and Setup. The key goes to the engine API, which has
    // the training engine check it with Hugging Face and save it in Hugging Face's own token file; it is
    // never shown again, logged, or kept by this app. hasKey offers "Remove saved key". True when a key
    // was saved or removed.
    winrt::Windows::Foundation::IAsyncOperation<bool> AskForHfKeyAsync(winrt::Microsoft::UI::Xaml::XamlRoot root,
                                                                       winrt::hstring reason, bool hasKey);

    // ---- engines a page depends on ---------------------------------------------------------------
    // Shows bar (with its Start action button) when any listed engine is not answering; hides it
    // otherwise. Returns true when every listed engine answers.
    bool UpdateEngineBar(winrt::Microsoft::UI::Xaml::Controls::InfoBar const& bar, std::vector<std::wstring> const& engineIds);
    // starts the listed engines one after another (the engine API first if listed)
    winrt::Windows::Foundation::IAsyncAction StartEnginesAsync(std::vector<std::wstring> engineIds);

    // ---- shell ------------------------------------------------------------------------------------------
    void OpenWithShell(std::wstring const& path);   // default app for a file or folder
    void ShowInExplorer(std::wstring const& path);  // Explorer window with the file selected

    // short local time from an ISO-8601 UTC timestamp ("2026-10-07T04:39:27+00:00" -> "Oct 7, 12:39 AM")
    winrt::hstring FriendlyTime(winrt::hstring const& iso);
}
