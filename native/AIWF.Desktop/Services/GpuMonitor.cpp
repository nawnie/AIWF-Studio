// GPU monitor implementation (sources and reasons are in GpuMonitor.h).
#include "pch.h"
#include <dxgi1_6.h>
#include "Services/GpuMonitor.h"

namespace aiwf
{
    // ---- the NVML C API, declared here so no NVIDIA SDK is needed to build ------------------------
    // Signatures follow nvml.h; nvmlReturn_t is an int where 0 means success.
    namespace nvml
    {
        using Return = int;
        using Device = void*;
        struct Memory { unsigned long long total, free, used; };
        struct Utilization { unsigned int gpu, memory; };
        constexpr int TemperatureGpu = 0;
        constexpr int ClockGraphics = 0;
    }

    struct GpuMonitor::NvmlApi
    {
        nvml::Return(__cdecl* init)() = nullptr;
        nvml::Return(__cdecl* deviceCount)(unsigned int*) = nullptr;
        nvml::Return(__cdecl* deviceByIndex)(unsigned int, nvml::Device*) = nullptr;
        nvml::Return(__cdecl* deviceName)(nvml::Device, char*, unsigned int) = nullptr;
        nvml::Return(__cdecl* memoryInfo)(nvml::Device, nvml::Memory*) = nullptr;
        nvml::Return(__cdecl* utilization)(nvml::Device, nvml::Utilization*) = nullptr;
        nvml::Return(__cdecl* temperature)(nvml::Device, int, unsigned int*) = nullptr;
        nvml::Return(__cdecl* powerUsage)(nvml::Device, unsigned int*) = nullptr;
        nvml::Return(__cdecl* powerLimit)(nvml::Device, unsigned int*) = nullptr;
        nvml::Return(__cdecl* clockInfo)(nvml::Device, int, unsigned int*) = nullptr;
        nvml::Return(__cdecl* driverVersion)(char*, unsigned int) = nullptr;
        nvml::Return(__cdecl* cudaVersion)(int*) = nullptr;
    };

    namespace
    {
        template <typename T>
        bool Resolve(HMODULE module, char const* name, T& target)
        {
            target = reinterpret_cast<T>(GetProcAddress(module, name));
            return target != nullptr;
        }

        std::wstring Widen(char const* text)
        {
            return std::wstring(winrt::to_hstring(std::string_view(text)));
        }

        // the LUID of the first NVIDIA adapter, formatted the way GPU counter instances name it
        std::wstring NvidiaLuidTag()
        {
            winrt::com_ptr<IDXGIFactory1> factory;
            if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(factory.put())))) return {};
            for (UINT index = 0;; ++index)
            {
                winrt::com_ptr<IDXGIAdapter1> adapter;
                if (factory->EnumAdapters1(index, adapter.put()) == DXGI_ERROR_NOT_FOUND) break;
                DXGI_ADAPTER_DESC1 desc{};
                if (FAILED(adapter->GetDesc1(&desc))) continue;
                if (desc.VendorId == 0x10DE && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE))
                {
                    wchar_t tag[64]{};
                    swprintf_s(tag, L"luid_0x%08x_0x%08x", static_cast<unsigned>(desc.AdapterLuid.HighPart), desc.AdapterLuid.LowPart);
                    return tag;
                }
            }
            return {};
        }
    }

    GpuMonitor& GpuMonitor::Instance()
    {
        static GpuMonitor* instance = new GpuMonitor();   // lives for the whole process
        return *instance;
    }

    // ---- one-time setup: load NVML, pick device 0, open the per-process counter ------------------
    void GpuMonitor::EnsureInitialized()
    {
        if (m_initialized) return;
        m_initialized = true;
        m_api = std::make_unique<NvmlApi>();

        m_nvml = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!m_nvml)
        {
            // older drivers kept NVML under Program Files
            wchar_t programFiles[MAX_PATH]{};
            if (ExpandEnvironmentStringsW(L"%ProgramW6432%\\NVIDIA Corporation\\NVSMI\\nvml.dll", programFiles, MAX_PATH))
            {
                m_nvml = LoadLibraryExW(programFiles, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
            }
        }
        if (!m_nvml)
        {
            m_initError = L"No NVIDIA driver telemetry (nvml.dll) was found on this PC.";
            return;
        }

        auto& api = *m_api;
        bool ok = Resolve(m_nvml, "nvmlInit_v2", api.init) && Resolve(m_nvml, "nvmlDeviceGetCount_v2", api.deviceCount) &&
                  Resolve(m_nvml, "nvmlDeviceGetHandleByIndex_v2", api.deviceByIndex) && Resolve(m_nvml, "nvmlDeviceGetName", api.deviceName) &&
                  Resolve(m_nvml, "nvmlDeviceGetMemoryInfo", api.memoryInfo) &&
                  Resolve(m_nvml, "nvmlDeviceGetUtilizationRates", api.utilization);
        // these are nice to have; the card still shows memory and load without them
        Resolve(m_nvml, "nvmlDeviceGetTemperature", api.temperature);
        Resolve(m_nvml, "nvmlDeviceGetPowerUsage", api.powerUsage);
        Resolve(m_nvml, "nvmlDeviceGetEnforcedPowerLimit", api.powerLimit);
        Resolve(m_nvml, "nvmlDeviceGetClockInfo", api.clockInfo);
        Resolve(m_nvml, "nvmlSystemGetDriverVersion", api.driverVersion);
        Resolve(m_nvml, "nvmlSystemGetCudaDriverVersion_v2", api.cudaVersion);
        if (!ok || api.init() != 0)
        {
            m_initError = L"The NVIDIA driver's telemetry library did not start.";
            return;
        }
        unsigned int count = 0;
        if (api.deviceCount(&count) != 0 || count == 0 || api.deviceByIndex(0, reinterpret_cast<nvml::Device*>(&m_device)) != 0)
        {
            m_initError = L"No NVIDIA GPU was found.";
            m_device = nullptr;
            return;
        }

        // this block reads the facts that never change while the app runs
        char text[128]{};
        if (api.deviceName(m_device, text, sizeof(text)) == 0) m_name = Widen(text);
        if (api.driverVersion && api.driverVersion(text, sizeof(text)) == 0) m_driver = Widen(text);
        int cuda = 0;
        if (api.cudaVersion && api.cudaVersion(&cuda) == 0 && cuda > 0)
        {
            m_cuda = std::to_wstring(cuda / 1000) + L"." + std::to_wstring((cuda % 1000) / 10);
        }

        // per-process VRAM counter (wildcard instance: one per process and adapter)
        m_luidTag = NvidiaLuidTag();
        if (PdhOpenQueryW(nullptr, 0, &m_query) == ERROR_SUCCESS)
        {
            if (PdhAddEnglishCounterW(m_query, L"\\GPU Process Memory(*)\\Dedicated Usage", 0, &m_counter) != ERROR_SUCCESS)
            {
                PdhCloseQuery(m_query);
                m_query = nullptr;
                m_counter = nullptr;
            }
        }
    }

    GpuSample GpuMonitor::Sample()
    {
        std::lock_guard guard(m_lock);
        // Home and the window footer both ask every 2 s; one fresh reading serves both
        auto now = std::chrono::steady_clock::now();
        if (m_hasCache && now - m_cachedAt < std::chrono::milliseconds(900)) return m_cache;
        EnsureInitialized();
        GpuSample sample = TakeSample();
        m_cache = sample;
        m_cachedAt = now;
        m_hasCache = true;
        return sample;
    }

    GpuSample GpuMonitor::TakeSample()
    {
        GpuSample sample;
        if (!m_device)
        {
            sample.error = m_initError;
            return sample;
        }
        auto& api = *m_api;
        sample.name = m_name;
        sample.driver = m_driver;
        sample.cuda = m_cuda;

        nvml::Memory memory{};
        if (api.memoryInfo(m_device, &memory) == 0)
        {
            sample.totalBytes = memory.total;
            sample.usedBytes = memory.used;
        }
        nvml::Utilization utilization{};
        if (api.utilization(m_device, &utilization) == 0)
        {
            sample.gpuUtilPercent = utilization.gpu;
            sample.memoryUtilPercent = utilization.memory;
        }
        unsigned int value = 0;
        if (api.temperature && api.temperature(m_device, nvml::TemperatureGpu, &value) == 0) sample.temperatureC = value;
        if (api.powerUsage && api.powerUsage(m_device, &value) == 0) sample.powerWatts = value / 1000.0;
        if (api.powerLimit && api.powerLimit(m_device, &value) == 0) sample.powerLimitWatts = value / 1000.0;
        if (api.clockInfo && api.clockInfo(m_device, nvml::ClockGraphics, &value) == 0) sample.graphicsClockMHz = value;
        sample.available = sample.totalBytes > 0;

        SampleProcessVram(sample);
        return sample;
    }

    // ---- dedicated VRAM per process, from the Windows GPU counters --------------------------------
    void GpuMonitor::SampleProcessVram(GpuSample& sample)
    {
        if (!m_query || !m_counter) return;
        if (PdhCollectQueryData(m_query) != ERROR_SUCCESS) return;
        DWORD bytes = 0;
        DWORD count = 0;
        if (PdhGetFormattedCounterArrayW(m_counter, PDH_FMT_LARGE, &bytes, &count, nullptr) != static_cast<PDH_STATUS>(PDH_MORE_DATA)) return;
        std::vector<uint8_t> buffer(bytes);
        auto items = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(buffer.data());
        if (PdhGetFormattedCounterArrayW(m_counter, PDH_FMT_LARGE, &bytes, &count, items) != ERROR_SUCCESS) return;

        // this loop parses instance names like pid_12940_luid_0x00000000_0x000198F5_phys_0
        for (DWORD i = 0; i < count; ++i)
        {
            std::wstring name = items[i].szName ? items[i].szName : L"";
            std::transform(name.begin(), name.end(), name.begin(), ::towlower);
            if (name.rfind(L"pid_", 0) != 0) continue;
            if (!m_luidTag.empty() && name.find(m_luidTag) == std::wstring::npos) continue;
            auto pid = static_cast<DWORD>(std::wcstoul(name.c_str() + 4, nullptr, 10));
            if (items[i].FmtValue.CStatus == ERROR_SUCCESS && items[i].FmtValue.largeValue > 0)
            {
                sample.processVram[pid] += static_cast<uint64_t>(items[i].FmtValue.largeValue);
            }
        }
    }
}
