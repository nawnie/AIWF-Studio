// Projects page: create projects, set the one in focus, show its recorded activity.
#pragma once
#include "ProjectsPage.g.h"

namespace winrt::AiwfDesktop::implementation
{
    struct ProjectsPage : ProjectsPageT<ProjectsPage>
    {
        ProjectsPage();

        void OnNavigatedTo(Microsoft::UI::Xaml::Navigation::NavigationEventArgs const& args);
        void OnNavigatedFrom(Microsoft::UI::Xaml::Navigation::NavigationEventArgs const& args);

        void StartEngine_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void Create_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void NewName_KeyDown(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::Input::KeyRoutedEventArgs const& args);
        void ProjectList_SelectionChanged(Windows::Foundation::IInspectable const& sender,
                                          Microsoft::UI::Xaml::Controls::SelectionChangedEventArgs const& args);

    private:
        fire_and_forget LoadAsync();
        fire_and_forget CreateAsync();
        fire_and_forget ShowDetailAsync(std::wstring id, std::wstring name);
        void UpdateEngineState();
        void SyncProject();
        void ApplyLayout(double width);

        // below this page width the project list sits above the activity details
        static constexpr double StackBelowWidth = 820;
        bool m_stacked = false;
        bool m_layoutKnown = false;

        bool m_loaded = false;
        bool m_suppressSelection = false;
        winrt::event_token m_statusToken{};
        winrt::event_token m_projectToken{};
        uint64_t m_detailGeneration = 0;
    };
}

namespace winrt::AiwfDesktop::factory_implementation
{
    struct ProjectsPage : ProjectsPageT<ProjectsPage, implementation::ProjectsPage>
    {
    };
}
