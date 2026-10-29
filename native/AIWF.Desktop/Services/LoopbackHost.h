#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace aiwf::http
{
    namespace detail
    {
        inline bool IsLocalhostName(std::wstring_view host) noexcept
        {
            constexpr std::wstring_view expected = L"localhost";
            if (host.size() != expected.size()) return false;
            for (size_t index = 0; index < host.size(); ++index)
            {
                wchar_t value = host[index];
                if (value >= L'A' && value <= L'Z') value = static_cast<wchar_t>(value + (L'a' - L'A'));
                if (value != expected[index]) return false;
            }
            return true;
        }

        inline bool IsLoopbackIpv4(std::wstring_view host) noexcept
        {
            unsigned octetIndex = 0;
            unsigned firstOctet = 0;
            size_t start = 0;
            while (start <= host.size())
            {
                const size_t end = host.find(L'.', start);
                const size_t stop = end == std::wstring_view::npos ? host.size() : end;
                const size_t length = stop - start;
                if (length == 0 || length > 3 || (length > 1 && host[start] == L'0')) return false;

                unsigned value = 0;
                for (size_t index = start; index < stop; ++index)
                {
                    const wchar_t character = host[index];
                    if (character < L'0' || character > L'9') return false;
                    value = value * 10 + static_cast<unsigned>(character - L'0');
                }
                if (value > 255 || octetIndex >= 4) return false;
                if (octetIndex == 0) firstOctet = value;
                ++octetIndex;

                if (end == std::wstring_view::npos) break;
                start = end + 1;
            }
            return octetIndex == 4 && firstOctet == 127;
        }

        inline bool ParseIpv6Groups(std::wstring_view text, std::array<uint16_t, 8>& groups, size_t& count) noexcept
        {
            count = 0;
            if (text.empty()) return true;
            size_t start = 0;
            while (start <= text.size())
            {
                const size_t end = text.find(L':', start);
                const size_t stop = end == std::wstring_view::npos ? text.size() : end;
                if (stop == start || stop - start > 4 || count >= groups.size()) return false;

                uint16_t value = 0;
                for (size_t index = start; index < stop; ++index)
                {
                    const wchar_t character = text[index];
                    unsigned digit = 0;
                    if (character >= L'0' && character <= L'9') digit = static_cast<unsigned>(character - L'0');
                    else if (character >= L'a' && character <= L'f') digit = static_cast<unsigned>(character - L'a' + 10);
                    else if (character >= L'A' && character <= L'F') digit = static_cast<unsigned>(character - L'A' + 10);
                    else return false;
                    value = static_cast<uint16_t>(value * 16 + digit);
                }
                groups[count++] = value;

                if (end == std::wstring_view::npos) break;
                start = end + 1;
            }
            return true;
        }

        inline bool IsLoopbackIpv6(std::wstring_view host) noexcept
        {
            if (host.find(L'.') != std::wstring_view::npos || host.find(L'%') != std::wstring_view::npos) return false;
            std::array<uint16_t, 8> address{};
            const size_t compression = host.find(L"::");
            if (compression == std::wstring_view::npos)
            {
                size_t count = 0;
                if (!ParseIpv6Groups(host, address, count) || count != address.size()) return false;
            }
            else
            {
                if (host.find(L"::", compression + 2) != std::wstring_view::npos) return false;
                std::array<uint16_t, 8> left{};
                std::array<uint16_t, 8> right{};
                size_t leftCount = 0;
                size_t rightCount = 0;
                if (!ParseIpv6Groups(host.substr(0, compression), left, leftCount) ||
                    !ParseIpv6Groups(host.substr(compression + 2), right, rightCount) || leftCount + rightCount >= address.size())
                    return false;
                for (size_t index = 0; index < leftCount; ++index) address[index] = left[index];
                for (size_t index = 0; index < rightCount; ++index) address[address.size() - rightCount + index] = right[index];
            }

            for (size_t index = 0; index + 1 < address.size(); ++index)
            {
                if (address[index] != 0) return false;
            }
            return address.back() == 1;
        }
    }

    inline bool IsLoopbackHost(std::wstring_view host) noexcept
    {
        if (detail::IsLocalhostName(host)) return true;
        if (!host.empty() && (host.front() == L'[' || host.back() == L']'))
        {
            if (host.size() < 3 || host.front() != L'[' || host.back() != L']') return false;
            host.remove_prefix(1);
            host.remove_suffix(1);
        }
        else if (host.find(L'[') != std::wstring_view::npos || host.find(L']') != std::wstring_view::npos)
        {
            return false;
        }
        return host.find(L':') == std::wstring_view::npos ? detail::IsLoopbackIpv4(host) : detail::IsLoopbackIpv6(host);
    }
}
