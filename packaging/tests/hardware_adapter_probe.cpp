// Exercises the same Windows adapter scan and fallback descriptions used by setup.
#include "Services/HardwareAdapters.h"
#include <iostream>

int wmain()
{
    using aiwf::hardware::DisplayAdapter;
    using aiwf::hardware::DescribeDisplayAdapter;
    constexpr uint64_t GiB = 1024ULL * 1024 * 1024;
    auto discrete = DescribeDisplayAdapter(DisplayAdapter{L"Synthetic AMD GPU", 8 * GiB, 16 * GiB, 0x1002});
    auto integrated = DescribeDisplayAdapter(DisplayAdapter{L"Synthetic Intel GPU", 0, 8 * GiB, 0x8086});
    auto unnamed = DescribeDisplayAdapter(DisplayAdapter{});
    using aiwf::hardware::ClassifyCudaDriver;
    using aiwf::hardware::CudaDriverLevel;
    using aiwf::hardware::ClassifyCudaComputeCapability;
    using aiwf::hardware::CudaComputeLevel;
    using aiwf::hardware::MaySuggestBundledSdxl;
    if (discrete.find(L"8.0 GiB dedicated video memory") == std::wstring::npos ||
        discrete.find(L"up to 16.0 GiB shared system memory (availability varies)") == std::wstring::npos ||
        integrated.find(L"dedicated video memory not reported") == std::wstring::npos ||
        unnamed.find(L"Unnamed graphics adapter") == std::wstring::npos ||
        ClassifyCudaDriver(L"12.9") != CudaDriverLevel::OlderThan13 ||
        ClassifyCudaDriver(L"13.0") != CudaDriverLevel::Reports13OrLater ||
        ClassifyCudaDriver(L"14.1") != CudaDriverLevel::Reports13OrLater ||
        ClassifyCudaDriver(L"") != CudaDriverLevel::Unknown ||
        ClassifyCudaDriver(L"13.") != CudaDriverLevel::Unknown ||
        ClassifyCudaDriver(L"13.4x") != CudaDriverLevel::Unknown ||
        ClassifyCudaComputeCapability(6, 1) != CudaComputeLevel::OlderThan75 ||
        ClassifyCudaComputeCapability(7, 0) != CudaComputeLevel::OlderThan75 ||
        ClassifyCudaComputeCapability(7, 5) != CudaComputeLevel::Reports75OrLater ||
        ClassifyCudaComputeCapability(8, 9) != CudaComputeLevel::Reports75OrLater ||
        ClassifyCudaComputeCapability(12, 0) != CudaComputeLevel::Reports75OrLater ||
        ClassifyCudaComputeCapability(-1, -1) != CudaComputeLevel::Unknown ||
        ClassifyCudaComputeCapability(0, 0) != CudaComputeLevel::Unknown ||
        ClassifyCudaComputeCapability(8, 10) != CudaComputeLevel::Unknown ||
        !MaySuggestBundledSdxl(true, 8 * GiB, L"13.0", 8, 9, 12 * GiB) ||
        MaySuggestBundledSdxl(false, 8 * GiB, L"13.0", 8, 9, 12 * GiB) ||
        MaySuggestBundledSdxl(true, 8 * GiB - 1, L"13.0", 8, 9, 12 * GiB) ||
        MaySuggestBundledSdxl(true, 8 * GiB, L"12.9", 8, 9, 12 * GiB) ||
        MaySuggestBundledSdxl(true, 8 * GiB, L"", 8, 9, 12 * GiB) ||
        MaySuggestBundledSdxl(true, 8 * GiB, L"13.0", 6, 1, 12 * GiB) ||
        MaySuggestBundledSdxl(true, 8 * GiB, L"13.0", -1, -1, 12 * GiB) ||
        MaySuggestBundledSdxl(true, 8 * GiB, L"13.0", 8, 9, 4 * GiB) ||
        MaySuggestBundledSdxl(true, 8 * GiB, L"13.0", 8, 9, 12 * GiB - 1) ||
        MaySuggestBundledSdxl(true, 8 * GiB, L"13.0", 8, 9, 0))
        return 1;

    auto adapters = aiwf::hardware::EnumerateDisplayAdapters();
    std::wcout << L"Descriptions, CUDA capability, and recommendation checks: passed\n";
    std::wcout << L"Hardware adapters reported by Windows: " << adapters.size() << L"\n";
    for (auto const& adapter : adapters)
        std::wcout << DescribeDisplayAdapter(adapter) << L"; vendor " << adapter.vendorId << L"\n";
    return 0;
}
