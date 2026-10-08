// GPU monitor: live telemetry for the NVIDIA card, and how much VRAM each process holds.
//
// Two sources, each used for what it reports reliably on Windows:
//   - NVML (nvml.dll, installed with the NVIDIA driver): name, driver and CUDA version, memory
//     used/total, utilization, temperature, power. Loaded at runtime, so the app still starts on
//     a PC without an NVIDIA driver; it then says so instead of showing numbers.
//   - Windows performance counters "GPU Process Memory(*)\Dedicated Usage": dedicated VRAM per
//     process ID. NVML's own per-process numbers are not available under WDDM, so the counters
//     are filtered to the NVIDIA adapter (matched by its LUID through DXGI).
// The Home page joins these per-process numbers with the engine supervisor's process lists to
// show which engine holds how much memory.
#pragma once

namespace aiwf
{
    struct GpuSample
    {
        bool available = false;
        std::wstring error;                 // why there are no numbers (no NVIDIA driver, ...)
        std::wstring name;                  // e.g. NVIDIA GeForce RTX 4070 Ti SUPER
        std::wstring driver;                // e.g. 616.92
        std::wstring cuda;                  // highest CUDA version the driver supports, e.g. 13.0
        uint64_t totalBytes = 0;
        uint64_t usedBytes = 0;
        uint32_t gpuUtilPercent = 0;
        uint32_t memoryUtilPercent = 0;
        uint32_t temperatureC = 0;
        double powerWatts = 0;
        double powerLimitWatts = 0;
        uint32_t graphicsClockMHz = 0;
        std::map<DWORD, uint64_t> processVram;   // dedicated VRAM by process ID, this GPU only
    };

    class GpuMonitor
    {
    public:
        static GpuMonitor& Instance();
        // takes a few milliseconds; call it from a background thread
        GpuSample Sample();

    private:
        GpuMonitor() = default;
        void EnsureInitialized();
        GpuSample TakeSample();
        void SampleProcessVram(GpuSample& sample);

        std::mutex m_lock;
        GpuSample m_cache;
        std::chrono::steady_clock::time_point m_cachedAt{};
        bool m_hasCache = false;
        bool m_initialized = false;
        std::wstring m_initError;
        HMODULE m_nvml = nullptr;
        void* m_device = nullptr;
        std::wstring m_name, m_driver, m_cuda;
        std::wstring m_luidTag;     // "luid_0x00000000_0x000198f5", lower case, for counter instance names
        PDH_HQUERY m_query = nullptr;
        PDH_HCOUNTER m_counter = nullptr;

        // NVML entry points (C API, resolved with GetProcAddress)
        struct NvmlApi;
        std::unique_ptr<NvmlApi> m_api;
    };
}
