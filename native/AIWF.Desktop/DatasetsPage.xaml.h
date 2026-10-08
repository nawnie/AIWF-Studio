// Datasets page: package revisions for training, and cataloging recent images.
#pragma once
#include "DatasetsPage.g.h"

namespace winrt::AiwfDesktop::implementation
{
    struct DatasetsPage : DatasetsPageT<DatasetsPage>
    {
        DatasetsPage();

        void OnNavigatedTo(Microsoft::UI::Xaml::Navigation::NavigationEventArgs const& args);
        void OnNavigatedFrom(Microsoft::UI::Xaml::Navigation::NavigationEventArgs const& args);

        void Refresh_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void StartEngine_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        fire_and_forget Catalog_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void Outputs_SelectionChanged(Windows::Foundation::IInspectable const& sender,
                                      Microsoft::UI::Xaml::Controls::SelectionChangedEventArgs const& args);

    private:
        fire_and_forget LoadAsync();
        void UpdateEngineState();
        void UpdateCatalogButton();

        bool m_loaded = false;
        bool m_loading = false;
        winrt::event_token m_statusToken{};
    };
}

namespace winrt::AiwfDesktop::factory_implementation
{
    struct DatasetsPage : DatasetsPageT<DatasetsPage, implementation::DatasetsPage>
    {
    };
}
