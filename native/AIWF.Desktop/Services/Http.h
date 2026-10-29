// Loopback HTTP for the engines: JSON in and out, raw bytes, and server-sent-event streams.
//
// Every engine listens on 127.0.0.1 (or another 127.x address), so these helpers refuse any
// other host before a request is made. Errors surface as winrt::hresult_error:
//   - an HTTP error status N becomes HRESULT 0x8019xxxx (FACILITY_HTTP, the same scheme as
//     Windows' HTTP_E_STATUS_* codes), and the message is the engine's own plain-text reason
//     taken from {"detail": {"code", "message"}} or {"detail": "..."} when it sent one;
//   - "nobody is listening" keeps the WinRT network HRESULT and gets a friendly message.
#pragma once

namespace aiwf::http
{
    // this section maps HTTP status codes to HRESULTs and back
    inline winrt::hresult StatusHresult(int status) noexcept
    {
        return MAKE_HRESULT(SEVERITY_ERROR, FACILITY_HTTP, static_cast<uint32_t>(status) & 0xFFFF);
    }
    inline int StatusOf(winrt::hresult_error const& error) noexcept
    {
        auto hr = static_cast<HRESULT>(error.code());
        return HRESULT_FACILITY(hr) == FACILITY_HTTP ? HRESULT_CODE(hr) : 0;
    }
    // true when the failure means "the engine is not running" rather than "the engine said no"
    bool IsNoAnswer(winrt::hresult_error const& error) noexcept;

    // this section is the request API used by every page and service
    winrt::Windows::Foundation::IAsyncOperation<winrt::Windows::Data::Json::JsonObject>
        GetJsonAsync(winrt::hstring url, winrt::hstring bearer = {}, uint32_t timeoutMs = 15000);

    winrt::Windows::Foundation::IAsyncOperation<winrt::Windows::Data::Json::JsonObject>
        PostJsonAsync(winrt::hstring url, winrt::Windows::Data::Json::JsonObject body,
                      winrt::hstring bearer = {}, uint32_t timeoutMs = 120000);

    winrt::Windows::Foundation::IAsyncOperation<winrt::Windows::Storage::Streams::IBuffer>
        GetBytesAsync(winrt::hstring url, uint32_t timeoutMs = 30000);

    // Probe: true when anything answers HTTP at url (any status, even 401 or 404). Used for
    // engine health, where a protected engine that says "unauthorized" is still running.
    winrt::Windows::Foundation::IAsyncOperation<bool> AnswersAsync(winrt::hstring url, uint32_t timeoutMs = 1500);
    winrt::Windows::Foundation::IAsyncOperation<bool> ReadyAsync(winrt::hstring url, uint32_t timeoutMs = 1500, winrt::hstring bearer = {});

    // Lets the UI stop a stream that is waiting on the network (Stop button in Chat).
    struct StreamCancel
    {
        std::atomic_bool cancelled{ false };
        void Cancel();
        void Track(winrt::Windows::Foundation::IAsyncInfo const& operation);
    private:
        std::mutex m_lock;
        winrt::Windows::Foundation::IAsyncInfo m_current{ nullptr };
    };

    // POST body, then call onEvent for each "data: {json}" line of a text/event-stream reply
    // until "data: [DONE]", the server closes the stream, onEvent returns false, or cancel fires.
    // onEvent runs on a background thread; marshal to the UI thread before touching XAML.
    winrt::Windows::Foundation::IAsyncOperation<bool> StreamSseAsync(
        winrt::hstring url,
        winrt::Windows::Data::Json::JsonObject body,
        winrt::hstring bearer,
        std::function<bool(winrt::Windows::Data::Json::JsonObject const&)> onEvent,
        std::shared_ptr<StreamCancel> cancel);

    // small JSON readers that never throw on a missing or mistyped field
    winrt::hstring Str(winrt::Windows::Data::Json::JsonObject const& obj, wchar_t const* key, wchar_t const* fallback = L"");
    double Num(winrt::Windows::Data::Json::JsonObject const& obj, wchar_t const* key, double fallback = 0);
    bool Bool(winrt::Windows::Data::Json::JsonObject const& obj, wchar_t const* key, bool fallback = false);
    winrt::Windows::Data::Json::JsonObject Obj(winrt::Windows::Data::Json::JsonObject const& obj, wchar_t const* key);
    winrt::Windows::Data::Json::JsonArray Arr(winrt::Windows::Data::Json::JsonObject const& obj, wchar_t const* key);

    // percent-encode one URL path segment (project names never go into paths, IDs do)
    winrt::hstring EscapeSegment(winrt::hstring const& value);
}
