// Read Windows display-adapter identity without loading an inference backend.
#pragma once

#include <Windows.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <cstdint>
#include <cwchar>
#include <string>
#include <vector>

namespace aiwf::hardware
{
    enum class CudaDriverLevel { Unknown, OlderThan13, Reports13OrLater };

    // NVML reports a major.minor version; this is a driver capability hint,
    // not proof that the GPU or the bundled runtime will run a particular job.
    inline CudaDriverLevel ClassifyCudaDriver(std::wstring const& version)
    {
        auto dot = version.find(L'.');
        if (dot == std::wstring::npos || dot == 0 || dot + 1 == version.size())
            return CudaDriverLevel::Unknown;
        unsigned int major = 0;
        for (size_t i = 0; i < version.size(); ++i)
        {
            if (i == dot) continue;
            if (version[i] < L'0' || version[i] > L'9') return CudaDriverLevel::Unknown;
            if (i < dot)
            {
                if (major > 1000) return CudaDriverLevel::Unknown;
                major = major * 10 + static_cast<unsigned int>(version[i] - L'0');
            }
        }
        if (major == 0 || major > 100) return CudaDriverLevel::Unknown;
        return major >= 13 ? CudaDriverLevel::Reports13OrLater : CudaDriverLevel::OlderThan13;
    }

    enum class CudaComputeLevel { Unknown, OlderThan75, Reports75OrLater };

    // The bundled CUDA 13 engine has no kernels for GPU architectures below 7.5.
    // Meeting this minimum alone does not guarantee the engine or model will run.
    inline CudaComputeLevel ClassifyCudaComputeCapability(int major, int minor)
    {
        if (major <= 0 || major > 100 || minor < 0 || minor > 9) return CudaComputeLevel::Unknown;
        return major > 7 || (major == 7 && minor >= 5)
            ? CudaComputeLevel::Reports75OrLater : CudaComputeLevel::OlderThan75;
    }

    inline bool MaySuggestBundledSdxl(bool telemetryAvailable, uint64_t totalBytes,
        std::wstring const& cudaDriver, int computeMajor, int computeMinor, uint64_t systemMemoryBytes)
    {
        constexpr uint64_t minimumVram = 8ULL * 1024 * 1024 * 1024;
        // Conservative onboarding threshold for a 6.94 GB checkpoint, not a guarantee of fit.
        constexpr uint64_t minimumSystemRam = 12ULL * 1024 * 1024 * 1024;
        return telemetryAvailable && totalBytes >= minimumVram && systemMemoryBytes >= minimumSystemRam &&
            ClassifyCudaDriver(cudaDriver) == CudaDriverLevel::Reports13OrLater &&
            ClassifyCudaComputeCapability(computeMajor, computeMinor) == CudaComputeLevel::Reports75OrLater;
    }

    struct DisplayAdapter
    {
        std::wstring name;
        uint64_t dedicatedBytes = 0;
        uint64_t sharedBytes = 0;
        uint32_t vendorId = 0;
    };

    inline std::vector<DisplayAdapter> EnumerateDisplayAdapters()
    {
        std::vector<DisplayAdapter> result;
        Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
        if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1),
                reinterpret_cast<void**>(factory.GetAddressOf())))) return result;

        for (UINT index = 0; index < 64; ++index)
        {
            Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
            HRESULT hr = factory->EnumAdapters1(index, adapter.GetAddressOf());
            if (hr == DXGI_ERROR_NOT_FOUND || FAILED(hr)) break;
            DXGI_ADAPTER_DESC1 desc{};
            if (FAILED(adapter->GetDesc1(&desc)) || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
            result.push_back({ desc.Description, static_cast<uint64_t>(desc.DedicatedVideoMemory),
                static_cast<uint64_t>(desc.SharedSystemMemory), desc.VendorId });
        }
        return result;
    }

    inline std::wstring FormatGiB(uint64_t bytes)
    {
        wchar_t text[48]{};
        swprintf_s(text, L"%.1f GiB", static_cast<double>(bytes) / 1073741824.0);
        return text;
    }

    inline std::wstring DescribeDisplayAdapter(DisplayAdapter const& adapter)
    {
        std::wstring text = adapter.name.empty() ? L"Unnamed graphics adapter" : adapter.name;
        if (adapter.dedicatedBytes)
            text += L"  ·  " + FormatGiB(adapter.dedicatedBytes) + L" dedicated video memory";
        else
            text += L"  ·  dedicated video memory not reported";
        if (adapter.sharedBytes)
            text += L"  ·  up to " + FormatGiB(adapter.sharedBytes) + L" shared system memory (availability varies)";
        return text;
    }
}
