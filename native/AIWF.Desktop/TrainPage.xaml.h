// Train page: base model (weights, download, key), package import, dry-run plan.
#pragma once
#include "TrainPage.g.h"

namespace winrt::AiwfDesktop::implementation
{
    struct TrainPage : TrainPageT<TrainPage>
    {
        TrainPage();

        void OnNavigatedTo(Microsoft::UI::Xaml::Navigation::NavigationEventArgs const& args);
        void OnNavigatedFrom(Microsoft::UI::Xaml::Navigation::NavigationEventArgs const& args);

        void StartEngine_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        fire_and_forget Key_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        fire_and_forget CancelDownload_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        fire_and_forget Import_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        fire_and_forget Plan_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);

    private:
        // one base model as the training engine reports it
        struct Model
        {
            std::wstring id;
            std::wstring label;
            std::wstring repo;
            double sizeB = 0;
            bool present = false;
            bool textCapable = true;
            std::wstring access;        // open, gated_ok, gated_no_key, gated_no_access, unavailable, unknown
            uint64_t downloadBytes = 0;
            std::wstring message;
        };

        fire_and_forget LoadModelsAsync(bool refresh);
        fire_and_forget LoadPackagesAsync();
        fire_and_forget DownloadAsync(Model model);
        void ShowModels();
        void UpdateEngineState();
        void UpdateProjectState();
        Windows::Foundation::IAsyncOperation<bool> AskForKeyAsync(hstring reason);

        std::vector<Model> m_models;
        std::wstring m_selectedModel;
        bool m_hasKey = false;
        std::wstring m_downloadJob;
        std::wstring m_datasetId;          // sha256-<manifest> after a successful import
        std::wstring m_datasetSha;
        std::wstring m_datasetName;
        std::wstring m_projectId;
        uint64_t m_projectGeneration = 0;
        bool m_importing = false;
        bool m_planning = false;
        bool m_loaded = false;
        winrt::event_token m_statusToken{};
        winrt::event_token m_projectToken{};
    };
}

namespace winrt::AiwfDesktop::factory_implementation
{
    struct TrainPage : TrainPageT<TrainPage, implementation::TrainPage>
    {
    };
}
