// Loopback HTTP for the engines (see Http.h for the error model).
#include "pch.h"
#include "Services/Http.h"

using namespace winrt;
using namespace winrt::Windows::Foundation;
using namespace winrt::Windows::Data::Json;
using namespace winrt::Windows::Storage::Streams;
using namespace winrt::Windows::Web::Http;
using namespace winrt::Windows::Web::Http::Filters;

namespace aiwf::http
{
    namespace
    {
        // ---- the shared client -------------------------------------------------------------
        // One HttpClient for the whole app: it is safe for concurrent requests and keeps loopback
        // connections warm. It is intentionally never destroyed, so no WinRT object outlives the
        // apartment at process exit. No cache (engine state changes every second) and no redirects.
        HttpClient& Client()
        {
            // WinRT types cannot be created with new, so a small holder struct is leaked instead
            struct Holder
            {
                HttpClient client{ nullptr };
            };
            static Holder* holder = []
            {
                HttpBaseProtocolFilter filter;
                filter.AllowUI(false);
                filter.AllowAutoRedirect(false);
                filter.CacheControl().ReadBehavior(HttpCacheReadBehavior::NoCache);
                filter.CacheControl().WriteBehavior(HttpCacheWriteBehavior::NoCache);
                auto created = new Holder();
                created->client = HttpClient(filter);
                return created;
            }();
            return holder->client;
        }

        // ---- address and error helpers -----------------------------------------------------
        // engines live on this PC only: http://127.x.x.x, localhost or [::1]
        Uri LoopbackUri(hstring const& url)
        {
            Uri uri{ url };
            std::wstring host{ uri.Host() };
            bool loopback = host == L"localhost" || host.rfind(L"127.", 0) == 0 || host == L"[::1]" || host == L"::1";
            if (!loopback || uri.SchemeName() != L"http")
            {
                throw hresult_error(E_ACCESSDENIED, L"Refusing a non-loopback engine address: " + url);
            }
            return uri;
        }

        // the engine's own words from an error body, or a plain fallback
        hstring ErrorMessage(int status, hstring const& body)
        {
            JsonObject obj{ nullptr };
            if (JsonObject::TryParse(body, obj))
            {
                // FastAPI: {"detail": {"code", "message"}} or {"detail": "text"}; llama.cpp: {"error": {...}} or {"error": "text"}
                for (wchar_t const* key : { L"detail", L"error" })
                {
                    auto value = obj.TryLookup(key);
                    if (!value) continue;
                    if (value.ValueType() == JsonValueType::String && !value.GetString().empty()) return value.GetString();
                    if (value.ValueType() == JsonValueType::Object)
                    {
                        auto message = Str(value.GetObject(), L"message");
                        if (!message.empty()) return message;
                    }
                }
            }
            return L"The engine answered HTTP " + to_hstring(status) + L".";
        }

        // network failures get a sentence a person can act on; the HRESULT is kept for IsNoAnswer
        hresult_error NoAnswerError(Uri const& uri, hresult_error const& error)
        {
            if (IsNoAnswer(error))
            {
                return hresult_error(error.code(), L"Nothing is answering at " + uri.SchemeName() + L"://" + uri.Host() + L":" +
                                                   to_hstring(uri.Port()) + L". Start the engine first.");
            }
            return error;
        }

        // ---- sending with a timeout ----------------------------------------------------------
        // WinRT's HttpClient has no per-request timeout, so a thread-pool timer cancels the call
        IAsyncOperation<HttpResponseMessage> SendAsync(HttpRequestMessage request, uint32_t timeoutMs,
                                                       HttpCompletionOption option = HttpCompletionOption::ResponseContentRead)
        {
            auto uri = request.RequestUri();
            auto operation = Client().SendRequestAsync(request, option);
            auto timer = Windows::System::Threading::ThreadPoolTimer::CreateTimer(
                [operation](auto&&) { operation.Cancel(); }, std::chrono::milliseconds(timeoutMs));
            HttpResponseMessage response{ nullptr };
            try
            {
                response = co_await operation;
            }
            catch (hresult_canceled const&)
            {
                throw hresult_error(HRESULT_FROM_WIN32(ERROR_TIMEOUT),
                                    L"The engine at " + uri.Host() + L":" + to_hstring(uri.Port()) + L" did not answer within " +
                                        to_hstring(timeoutMs / 1000) + L" seconds.");
            }
            catch (hresult_error const& error)
            {
                timer.Cancel();
                throw NoAnswerError(uri, error);
            }
            timer.Cancel();
            co_return response;
        }

        // reads a reply: JSON object on success, hresult_error with the engine's reason otherwise
        IAsyncOperation<JsonObject> ReadJsonAsync(HttpResponseMessage response)
        {
            auto status = static_cast<int>(response.StatusCode());
            auto text = co_await response.Content().ReadAsStringAsync();
            if (!response.IsSuccessStatusCode())
            {
                throw hresult_error(StatusHresult(status), ErrorMessage(status, text));
            }
            JsonObject obj{ nullptr };
            if (!JsonObject::TryParse(text, obj))
            {
                throw hresult_error(StatusHresult(502), L"The engine sent a reply that is not JSON.");
            }
            co_return obj;
        }

        HttpRequestMessage NewRequest(HttpMethod const& method, hstring const& url, hstring const& bearer, wchar_t const* accept)
        {
            HttpRequestMessage request(method, LoopbackUri(url));
            request.Headers().Accept().TryParseAdd(accept);
            if (!bearer.empty())
            {
                request.Headers().Authorization(Headers::HttpCredentialsHeaderValue(L"Bearer", bearer));
            }
            return request;
        }
    }

    // ---- error classification ------------------------------------------------------------------
    bool IsNoAnswer(hresult_error const& error) noexcept
    {
        auto hr = static_cast<HRESULT>(error.code());
        if (HRESULT_FACILITY(hr) != FACILITY_WIN32) return false;
        switch (HRESULT_CODE(hr))
        {
        case 12029:  // ERROR_WINHTTP_CANNOT_CONNECT: nobody listening
        case 12030:  // connection aborted
        case 12031:  // connection reset
        case 12002:  // WinHTTP timeout
        case 12007:  // name not resolved
        case ERROR_CONNECTION_REFUSED:
        case ERROR_TIMEOUT:
            return true;
        default:
            return false;
        }
    }

    // ---- public request API -----------------------------------------------------------------------
    IAsyncOperation<JsonObject> GetJsonAsync(hstring url, hstring bearer, uint32_t timeoutMs)
    {
        auto request = NewRequest(HttpMethod::Get(), url, bearer, L"application/json");
        auto response = co_await SendAsync(request, timeoutMs);
        co_return co_await ReadJsonAsync(response);
    }

    IAsyncOperation<JsonObject> PostJsonAsync(hstring url, JsonObject body, hstring bearer, uint32_t timeoutMs)
    {
        auto request = NewRequest(HttpMethod::Post(), url, bearer, L"application/json");
        request.Content(HttpStringContent(body.Stringify(), UnicodeEncoding::Utf8, L"application/json"));
        auto response = co_await SendAsync(request, timeoutMs);
        co_return co_await ReadJsonAsync(response);
    }

    IAsyncOperation<IBuffer> GetBytesAsync(hstring url, uint32_t timeoutMs)
    {
        auto request = NewRequest(HttpMethod::Get(), url, {}, L"*/*");
        auto response = co_await SendAsync(request, timeoutMs);
        if (!response.IsSuccessStatusCode())
        {
            auto status = static_cast<int>(response.StatusCode());
            auto text = co_await response.Content().ReadAsStringAsync();
            throw hresult_error(StatusHresult(status), ErrorMessage(status, text));
        }
        co_return co_await response.Content().ReadAsBufferAsync();
    }

    IAsyncOperation<bool> AnswersAsync(hstring url, uint32_t timeoutMs)
    {
        try
        {
            // the (small) body is read in full so the connection ends cleanly; dropping it after the
            // headers resets the socket, which asyncio-based engines log as an error on every probe
            auto request = NewRequest(HttpMethod::Get(), url, {}, L"*/*");
            auto response = co_await SendAsync(request, timeoutMs);
            response.Close();
            co_return true;   // any HTTP status means a server is there
        }
        catch (hresult_error const&)
        {
            co_return false;
        }
    }

    IAsyncOperation<bool> ReadyAsync(hstring url, uint32_t timeoutMs, hstring bearer)
    {
        try
        {
            auto request = NewRequest(HttpMethod::Get(), url, bearer, L"*/*");
            auto response = co_await SendAsync(request, timeoutMs);
            bool ready = response.IsSuccessStatusCode();
            response.Close();
            co_return ready;
        }
        catch (hresult_error const&)
        {
            co_return false;
        }
    }

    // ---- streaming (server-sent events) --------------------------------------------------------
    void StreamCancel::Cancel()
    {
        cancelled = true;
        std::lock_guard guard(m_lock);
        if (m_current) m_current.Cancel();
    }

    void StreamCancel::Track(IAsyncInfo const& operation)
    {
        std::lock_guard guard(m_lock);
        m_current = operation;
        if (cancelled) operation.Cancel();
    }

    IAsyncOperation<bool> StreamSseAsync(hstring url, JsonObject body, hstring bearer,
                                std::function<bool(JsonObject const&)> onEvent, std::shared_ptr<StreamCancel> cancel)
    {
        co_await resume_background();
        auto request = NewRequest(HttpMethod::Post(), url, bearer, L"text/event-stream");
        request.Content(HttpStringContent(body.Stringify(), UnicodeEncoding::Utf8, L"application/json"));

        // this block waits for the reply headers; Stop before they arrive simply ends the call
        auto send = Client().SendRequestAsync(request, HttpCompletionOption::ResponseHeadersRead);
        cancel->Track(send);
        HttpResponseMessage response{ nullptr };
        try
        {
            response = co_await send;
        }
        catch (hresult_canceled const&)
        {
            co_return false;
        }
        catch (hresult_error const& error)
        {
            throw NoAnswerError(request.RequestUri(), error);
        }
        if (!response.IsSuccessStatusCode())
        {
            auto status = static_cast<int>(response.StatusCode());
            auto text = co_await response.Content().ReadAsStringAsync();
            throw hresult_error(StatusHresult(status), ErrorMessage(status, text));
        }
        // A successful response header means the engine accepted the submitted question.

        // this loop reads raw bytes and splits complete lines before decoding, so a UTF-8
        // character split across two network reads is never decoded half-way
        auto stream = co_await response.Content().ReadAsInputStreamAsync();
        DataReader reader(stream);
        reader.InputStreamOptions(InputStreamOptions::Partial);
        std::string pending;
        bool finished = false;
        while (!finished && !cancel->cancelled)
        {
            auto load = reader.LoadAsync(16 * 1024);
            cancel->Track(load);
            uint32_t count = 0;
            try
            {
                count = co_await load;
            }
            catch (hresult_canceled const&)
            {
                break;
            }
            if (count == 0) break;   // the engine closed the stream

            std::vector<uint8_t> bytes(count);
            reader.ReadBytes(bytes);
            pending.append(bytes.begin(), bytes.end());

            size_t newline;
            while (!finished && (newline = pending.find('\n')) != std::string::npos)
            {
                std::string line = pending.substr(0, newline);
                pending.erase(0, newline + 1);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.rfind("data:", 0) != 0) continue;   // blank separators, comments, event names
                std::string_view payload(line);
                payload.remove_prefix(5);
                while (!payload.empty() && payload.front() == ' ') payload.remove_prefix(1);
                if (payload == "[DONE]")
                {
                    finished = true;
                    break;
                }
                JsonObject obj{ nullptr };
                if (!JsonObject::TryParse(to_hstring(payload), obj)) continue;
                if (!onEvent(obj)) finished = true;
            }
        }
        // closing the reply drops the connection, which tells llama.cpp to stop generating
        response.Close();
        co_return true;
    }

    // ---- tolerant JSON readers ---------------------------------------------------------------------
    hstring Str(JsonObject const& obj, wchar_t const* key, wchar_t const* fallback)
    {
        if (!obj) return fallback;
        auto value = obj.TryLookup(key);
        if (!value) return fallback;
        switch (value.ValueType())
        {
        case JsonValueType::String: return value.GetString();
        case JsonValueType::Number:
        {
            auto number = value.GetNumber();
            if (number == static_cast<double>(static_cast<int64_t>(number))) return to_hstring(static_cast<int64_t>(number));
            return to_hstring(number);
        }
        case JsonValueType::Boolean: return value.GetBoolean() ? L"true" : L"false";
        default: return fallback;
        }
    }

    double Num(JsonObject const& obj, wchar_t const* key, double fallback)
    {
        if (!obj) return fallback;
        auto value = obj.TryLookup(key);
        return value && value.ValueType() == JsonValueType::Number ? value.GetNumber() : fallback;
    }

    bool Bool(JsonObject const& obj, wchar_t const* key, bool fallback)
    {
        if (!obj) return fallback;
        auto value = obj.TryLookup(key);
        return value && value.ValueType() == JsonValueType::Boolean ? value.GetBoolean() : fallback;
    }

    JsonObject Obj(JsonObject const& obj, wchar_t const* key)
    {
        if (!obj) return nullptr;
        auto value = obj.TryLookup(key);
        return value && value.ValueType() == JsonValueType::Object ? value.GetObject() : nullptr;
    }

    JsonArray Arr(JsonObject const& obj, wchar_t const* key)
    {
        if (!obj) return JsonArray();
        auto value = obj.TryLookup(key);
        return value && value.ValueType() == JsonValueType::Array ? value.GetArray() : JsonArray();
    }

    hstring EscapeSegment(hstring const& value)
    {
        return Uri::EscapeComponent(value);
    }
}
