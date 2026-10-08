// Chat page: model picker, streaming replies with live speed, optional project context.
#pragma once
#include "ChatPage.g.h"
#include "Services/Http.h"

namespace winrt::AiwfDesktop::implementation
{
    struct ChatPage : ChatPageT<ChatPage>
    {
        ChatPage();

        void OnNavigatedTo(Microsoft::UI::Xaml::Navigation::NavigationEventArgs const& args);
        void OnNavigatedFrom(Microsoft::UI::Xaml::Navigation::NavigationEventArgs const& args);

        void NewChat_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void Send_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void Stop_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void StartEngine_Click(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::RoutedEventArgs const& args);
        void MessageInput_PreviewKeyDown(Windows::Foundation::IInspectable const& sender, Microsoft::UI::Xaml::Input::KeyRoutedEventArgs const& args);

    private:
        // one turn of the conversation as it is sent back to the model
        struct Turn
        {
            std::wstring role;      // user or assistant
            std::wstring content;
        };

        fire_and_forget LoadModelsAsync();
        fire_and_forget SendAsync();
        void UpdateEngineState();
        void SetStreaming(bool streaming);
        Microsoft::UI::Xaml::Controls::Border AddUserBubble(std::wstring const& text);
        void ScrollToEnd();

        std::vector<Turn> m_history;
        std::shared_ptr<aiwf::http::StreamCancel> m_cancel;
        bool m_streaming = false;
        bool m_modelsLoaded = false;
        winrt::event_token m_statusToken{};
    };
}

namespace winrt::AiwfDesktop::factory_implementation
{
    struct ChatPage : ChatPageT<ChatPage, implementation::ChatPage>
    {
    };
}
