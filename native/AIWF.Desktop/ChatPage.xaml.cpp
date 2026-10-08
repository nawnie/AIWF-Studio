// Chat page implementation: streaming chat against the chat engine's OpenAI-compatible API.
#include "pch.h"
#include "ChatPage.xaml.h"
#if __has_include("ChatPage.g.cpp")
#include "ChatPage.g.cpp"
#endif
#include "Services/AppState.h"
#include "Services/EngineSupervisor.h"
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
        // the same framing the engine API's qwen_ask uses, so the model sees context the same way
        constexpr wchar_t ContextPreamble[] =
            L"You are helping with an AIWF Studio project. The project context below was sent explicitly by the user. "
            L"It contains identifiers and counts only, not file contents.\n\n";

        // last part of a long text, so the "thinking" line shows what the model is on right now
        std::wstring Tail(std::wstring const& text, size_t count)
        {
            if (text.size() <= count) return text;
            auto tail = text.substr(text.size() - count);
            auto space = tail.find(L' ');
            return L"..." + (space != std::wstring::npos ? tail.substr(space + 1) : tail);
        }
    }

    ChatPage::ChatPage()
    {
        InitializeComponent();
        SetStreaming(false);
    }

    // ---- page lifetime ------------------------------------------------------------------------------------
    void ChatPage::OnNavigatedTo(NavigationEventArgs const&)
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
        UpdateEngineState();
        MessageInput().Focus(FocusState::Programmatic);
    }

    void ChatPage::OnNavigatedFrom(NavigationEventArgs const&)
    {
        aiwf::EngineSupervisor::Instance().StatusChanged(m_statusToken);
    }

    void ChatPage::UpdateEngineState()
    {
        bool ready = ui::UpdateEngineBar(EngineBar(), { L"chat" });
        SendButton().IsEnabled(ready && !m_streaming);
        if (ready && !m_modelsLoaded) LoadModelsAsync();
        if (!ready) m_modelsLoaded = false;
    }

    void ChatPage::StartEngine_Click(IInspectable const&, RoutedEventArgs const&)
    {
        ui::StartEnginesAsync({ L"chat" });
    }

    // ---- models: loaded ones first, so the default choice answers right away -----------------------------
    fire_and_forget ChatPage::LoadModelsAsync()
    {
        m_modelsLoaded = true;   // set first so the 2-second status refresh does not start a second load
        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        auto& state = aiwf::AppState::Get();
        JsonObject body{ nullptr };
        hstring failure;
        try
        {
            body = co_await http::GetJsonAsync(state.ChatApi() + L"/v1/models", state.ChatKey());
        }
        catch (hresult_error const& error)
        {
            failure = error.message();
        }
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile
        if (!failure.empty())
        {
            m_modelsLoaded = false;
            ChatStatus().Text(L"Could not list models: " + failure);
            co_return;
        }

        auto previous = ui::SelectedTag(ModelPicker());
        std::vector<std::pair<hstring, bool>> models;
        for (auto const& value : http::Arr(body, L"data"))
        {
            auto model = value.GetObject();
            auto status = http::Obj(model, L"status");
            models.emplace_back(http::Str(model, L"id"), http::Str(status, L"value") == L"loaded");
        }
        std::stable_sort(models.begin(), models.end(), [](auto const& a, auto const& b) { return a.second && !b.second; });

        ModelPicker().Items().Clear();
        int selected = -1;
        // this loop shows each model once, marking the ones already in GPU memory
        for (size_t i = 0; i < models.size(); ++i)
        {
            ComboBoxItem item;
            item.Content(box_value(models[i].second ? models[i].first + L"   ● loaded" : models[i].first));
            item.Tag(box_value(models[i].first));
            ModelPicker().Items().Append(item);
            if (models[i].first == previous) selected = static_cast<int>(i);
        }
        if (selected < 0 && !models.empty()) selected = 0;
        ModelPicker().SelectedIndex(selected);
        ChatStatus().Text(models.empty() ? L"The chat engine reports no models." :
                          hstring(std::to_wstring(models.size()) + L" models available on this PC."));
    }

    // ---- sending and streaming --------------------------------------------------------------------------------
    void ChatPage::Send_Click(IInspectable const&, RoutedEventArgs const&)
    {
        SendAsync();
    }

    void ChatPage::MessageInput_PreviewKeyDown(IInspectable const&, Microsoft::UI::Xaml::Input::KeyRoutedEventArgs const& args)
    {
        // Enter sends; Shift+Enter keeps the newline the text box would insert
        if (args.Key() == Windows::System::VirtualKey::Enter && !(GetKeyState(VK_SHIFT) & 0x8000))
        {
            args.Handled(true);
            SendAsync();
        }
    }

    void ChatPage::Stop_Click(IInspectable const&, RoutedEventArgs const&)
    {
        if (m_cancel) m_cancel->Cancel();
    }

    void ChatPage::NewChat_Click(IInspectable const&, RoutedEventArgs const&)
    {
        if (m_streaming) return;
        if (m_cancel) m_cancel->Cancel();
        m_history.clear();
        Messages().Children().Clear();
        Messages().Children().Append(EmptyState());
        ChatStatus().Text(L"New conversation.");
        MessageInput().Focus(FocusState::Programmatic);
    }

    fire_and_forget ChatPage::SendAsync()
    {
        if (m_streaming) co_return;
        std::wstring text{ MessageInput().Text() };
        while (!text.empty() && iswspace(text.back())) text.pop_back();
        while (!text.empty() && iswspace(text.front())) text.erase(0, 1);
        auto model = ui::SelectedTag(ModelPicker());
        if (text.empty()) co_return;
        if (model.empty())
        {
            ChatStatus().Text(L"Choose a model first.");
            co_return;
        }

        auto strong = get_strong();
        auto dispatcher = DispatcherQueue();
        auto& state = aiwf::AppState::Get();
        MessageInput().Text(L"");
        if (EmptyState().Parent()) Messages().Children().Clear();
        AddUserBubble(text);
        m_history.push_back({ L"user", text });

        // this block builds the model's bubble: thinking line, answer, and a stats line
        Border bubble;
        bubble.Style(ui::AppStyle(L"ModelBubbleStyle"));
        StackPanel parts;
        parts.Spacing(6);
        auto thinking = ui::Text(L"Waiting for the model (it loads into GPU memory on first use)...", L"CaptionTextBlockStyle", 0.6);
        thinking.FontStyle(Windows::UI::Text::FontStyle::Italic);
        thinking.MaxLines(3);
        thinking.TextTrimming(TextTrimming::CharacterEllipsis);
        auto answer = ui::Text(L"", L"BodyTextBlockStyle");
        answer.IsTextSelectionEnabled(true);
        answer.Visibility(Visibility::Collapsed);
        auto stats = ui::Text(L"", L"CaptionTextBlockStyle", 0.55);
        stats.Visibility(Visibility::Collapsed);
        parts.Children().Append(thinking);
        parts.Children().Append(answer);
        parts.Children().Append(stats);
        bubble.Child(parts);
        Messages().Children().Append(bubble);
        ScrollToEnd();
        m_cancel = std::make_shared<http::StreamCancel>();
        SetStreaming(true);

        // this block assembles the request: optional project context, then the whole conversation
        JsonArray messages;
        hstring failure;
        hstring historyWarning;
        hstring contextSha256;
        bool contextSent = false;
        auto projectId = state.ProjectId();
        if (ContextToggle().IsOn())
        {
            if (projectId.empty())
            {
                failure = L"Project context is on, but no project is chosen. Pick one in the title bar.";
            }
            else
            {
                try
                {
                    auto context = co_await http::GetJsonAsync(state.EngineApi() + L"/projects/" + http::EscapeSegment(hstring(projectId)) + L"/qwen-context");
                    JsonObject system;
                    system.Insert(L"role", JsonValue::CreateStringValue(L"system"));
                    system.Insert(L"content", JsonValue::CreateStringValue(hstring(ContextPreamble) + http::Str(context, L"context")));
                    messages.Append(system);
                    contextSha256 = http::Str(context, L"context_sha256");
                    contextSent = true;
                }
                catch (hresult_error const& error)
                {
                    failure = L"Project context could not be read, so nothing was sent: " + error.message();
                }
            }
        }
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile
        if (m_cancel->cancelled)
        {
            m_history.pop_back();
            SetStreaming(false);
            co_return;
        }
        if (!failure.empty())
        {
            thinking.Text(failure);
            thinking.FontStyle(Windows::UI::Text::FontStyle::Normal);
            m_history.pop_back();
            SetStreaming(false);
            co_return;
        }
        for (auto const& turn : m_history)
        {
            JsonObject message;
            message.Insert(L"role", JsonValue::CreateStringValue(turn.role));
            message.Insert(L"content", JsonValue::CreateStringValue(turn.content));
            messages.Append(message);
        }
        JsonObject body;
        body.Insert(L"model", JsonValue::CreateStringValue(model));
        body.Insert(L"stream", JsonValue::CreateBooleanValue(true));
        body.Insert(L"messages", messages);

        // shared between the network thread (writer) and this coroutine (reader once the stream ends)
        struct Progress
        {
            std::wstring content;
            std::wstring reasoning;
            int chunks = 0;
            double serverTokensPerSecond = 0;
            int serverTokens = 0;
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            std::chrono::steady_clock::time_point firstToken{};
            std::chrono::steady_clock::time_point answerStart{};
            std::chrono::steady_clock::time_point lastPaint{};
        };
        auto progress = std::make_shared<Progress>();
        auto weak = get_weak();

        // this callback runs on a network thread for every streamed event
        auto onEvent = [progress, dispatcher, thinking, answer, weak](JsonObject const& event) -> bool
        {
            auto now = std::chrono::steady_clock::now();
            auto choices = http::Arr(event, L"choices");
            if (choices.Size() > 0)
            {
                auto delta = http::Obj(choices.GetObjectAt(0), L"delta");
                auto piece = http::Str(delta, L"content");
                auto thought = http::Str(delta, L"reasoning_content");
                if (!piece.empty() || !thought.empty())
                {
                    if (progress->chunks == 0) progress->firstToken = now;
                    if (!piece.empty() && progress->content.empty()) progress->answerStart = now;
                    progress->content += piece;
                    progress->reasoning += thought;
                    ++progress->chunks;
                }
            }
            // llama.cpp puts its own measured speed in the final event
            if (auto timings = http::Obj(event, L"timings"))
            {
                progress->serverTokensPerSecond = http::Num(timings, L"predicted_per_second");
                progress->serverTokens = static_cast<int>(http::Num(timings, L"predicted_n"));
            }
            // repaint at most ~16 times a second; the final text is painted after the stream ends
            if (now - progress->lastPaint > std::chrono::milliseconds(60))
            {
                progress->lastPaint = now;
                std::wstring content = progress->content;
                std::wstring reasoning = progress->reasoning;
                dispatcher.TryEnqueue([weak, thinking, answer, content, reasoning]
                {
                    auto self = weak.get();
                    if (!self || aiwf::AppState::Get().ShuttingDown()) return;
                    if (content.empty())
                    {
                        if (!reasoning.empty()) thinking.Text(L"Thinking: " + Tail(reasoning, 280));
                    }
                    else
                    {
                        answer.Visibility(Visibility::Visible);
                        answer.Text(content);
                    }
                    self->ScrollToEnd();
                });
            }
            return true;
        };

        bool streamAccepted = false;
        try
        {
            streamAccepted = co_await http::StreamSseAsync(state.ChatApi() + L"/v1/chat/completions", body, state.ChatKey(), onEvent, m_cancel);
        }
        catch (hresult_error const& error)
        {
            failure = error.message();
        }
        co_await wil::resume_foreground(dispatcher);
        if (aiwf::AppState::Get().ShuttingDown()) co_return;   // the window closed meanwhile

        // Record after the engine accepts the stream. This records the submitted question, not an answer.
        if (contextSent && streamAccepted)
        {
            try
            {
                JsonObject record;
                record.Insert(L"model_id", JsonValue::CreateStringValue(model));
                record.Insert(L"question", JsonValue::CreateStringValue(hstring(text)));
                record.Insert(L"context_sha256", JsonValue::CreateStringValue(contextSha256));
                co_await http::PostJsonAsync(state.EngineApi() + L"/projects/" + http::EscapeSegment(hstring(projectId)) + L"/chat-question", record);
            }
            catch (hresult_error const& error)
            {
                historyWarning = L"The question was sent, but project history could not be updated: " + error.message();
            }
        }

        // this block paints the final state of the bubble
        bool stopped = m_cancel && m_cancel->cancelled;
        auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - progress->start).count();
        if (!progress->content.empty())
        {
            answer.Visibility(Visibility::Visible);
            answer.Text(progress->content);
        }
        if (!progress->reasoning.empty())
        {
            auto thought = progress->answerStart.time_since_epoch().count() != 0
                               ? std::chrono::duration<double>(progress->answerStart - progress->firstToken).count()
                               : elapsed;
            wchar_t line[64]{};
            swprintf_s(line, L"Thought for %.1f s", thought);
            thinking.Text(line);
            thinking.MaxLines(1);
        }
        else if (!progress->content.empty())
        {
            thinking.Visibility(Visibility::Collapsed);
        }
        if (!failure.empty())
        {
            thinking.Visibility(Visibility::Visible);
            thinking.FontStyle(Windows::UI::Text::FontStyle::Normal);
            thinking.Text(L"The chat engine could not answer: " + failure);
        }
        else if (progress->content.empty() && !stopped)
        {
            thinking.Text(progress->reasoning.empty() ? L"The model returned no text." : L"The model only produced thinking, no answer.");
        }

        // speed: llama.cpp's own measurement when it sent one, otherwise streamed pieces per second
        double tokensPerSecond = progress->serverTokensPerSecond;
        int tokens = progress->serverTokens;
        if (tokensPerSecond <= 0 && progress->chunks > 1)
        {
            auto generating = std::chrono::duration<double>(std::chrono::steady_clock::now() - progress->firstToken).count();
            tokens = progress->chunks;
            tokensPerSecond = generating > 0 ? progress->chunks / generating : 0;
        }
        if (progress->chunks > 0)
        {
            auto firstToken = std::chrono::duration<double>(progress->firstToken - progress->start).count();
            wchar_t line[200]{};
            swprintf_s(line, L"%s  ·  %.1f tokens/s  ·  %d tokens  ·  first token after %.1f s%s", model.c_str(), tokensPerSecond,
                       tokens, firstToken, stopped ? L"  ·  stopped" : L"");
            stats.Text(line);
            stats.Visibility(Visibility::Visible);
            ChatStatus().Text(line);
        }
        if (!progress->content.empty())
        {
            m_history.push_back({ L"assistant", progress->content });
        }
        else
        {
            m_history.pop_back();   // a failed turn is not sent again with the next message
        }
        SetStreaming(false);
        ScrollToEnd();
        if (!historyWarning.empty()) ChatStatus().Text(historyWarning);
        MessageInput().Focus(FocusState::Programmatic);
    }

    // ---- small helpers --------------------------------------------------------------------------------------------
    Border ChatPage::AddUserBubble(std::wstring const& text)
    {
        Border bubble;
        bubble.Style(ui::AppStyle(L"UserBubbleStyle"));
        auto body = ui::Text(hstring(text), L"BodyTextBlockStyle");
        body.IsTextSelectionEnabled(true);
        body.Foreground(Application::Current().Resources().Lookup(box_value(L"TextOnAccentFillColorPrimaryBrush")).as<Media::Brush>());
        bubble.Child(body);
        Messages().Children().Append(bubble);
        return bubble;
    }

    void ChatPage::SetStreaming(bool streaming)
    {
        m_streaming = streaming;
        SendButton().Visibility(streaming ? Visibility::Collapsed : Visibility::Visible);
        StopButton().Visibility(streaming ? Visibility::Visible : Visibility::Collapsed);
        ModelPicker().IsEnabled(!streaming);
        NewChatButton().IsEnabled(!streaming);
        SendButton().IsEnabled(!streaming && aiwf::EngineSupervisor::Instance().IsAnswering(L"chat"));
    }

    void ChatPage::ScrollToEnd()
    {
        // follow the stream only when the reader is already at the bottom
        auto scroll = MessagesScroll();
        bool atBottom = scroll.VerticalOffset() >= scroll.ScrollableHeight() - 120;
        scroll.UpdateLayout();
        if (atBottom) scroll.ChangeView(nullptr, scroll.ScrollableHeight(), nullptr, true);
    }
}
