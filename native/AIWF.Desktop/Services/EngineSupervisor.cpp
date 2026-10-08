// Engine supervisor implementation (ownership rules are in EngineSupervisor.h).
#include "pch.h"
// winsock2.h supplies AF_INET (WIN32_LEAN_AND_MEAN leaves it out); iphlpapi.h has the TCP owner tables
#include <winsock2.h>
#include <iphlpapi.h>
#include <tlhelp32.h>
#include "Services/AppState.h"
#include "Services/EngineSupervisor.h"
#include "Services/Http.h"
#include "Services/Paths.h"

using namespace winrt;
using namespace winrt::Windows::Foundation;
using namespace winrt::Windows::Data::Json;

namespace aiwf
{
    // ---- an engine this app started ------------------------------------------------------------
    // The job owns the whole process tree: TerminateJobObject (Stop) or closing the job handle
    // (app exit, thanks to KILL_ON_JOB_CLOSE) ends every process the engine spawned.
    struct EngineSupervisor::Owned
    {
        wil::unique_handle job;
        std::vector<wil::unique_handle> processes;
        std::wstring logPath;
    };

    namespace
    {
        // ---- process tree helpers ----------------------------------------------------------
        std::vector<DWORD> JobPids(HANDLE job)
        {
            std::vector<uint8_t> buffer(sizeof(JOBOBJECT_BASIC_PROCESS_ID_LIST) + sizeof(ULONG_PTR) * 512);
            auto list = reinterpret_cast<JOBOBJECT_BASIC_PROCESS_ID_LIST*>(buffer.data());
            if (!QueryInformationJobObject(job, JobObjectBasicProcessIdList, list, static_cast<DWORD>(buffer.size()), nullptr))
            {
                return {};
            }
            std::vector<DWORD> pids;
            for (DWORD i = 0; i < list->NumberOfProcessIdsInList; ++i)
            {
                pids.push_back(static_cast<DWORD>(list->ProcessIdList[i]));
            }
            return pids;
        }

        // parent -> children for every process on the machine, from one Toolhelp snapshot
        std::multimap<DWORD, DWORD> ProcessTree()
        {
            std::multimap<DWORD, DWORD> children;
            wil::unique_handle snapshot(CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0));
            if (!snapshot || snapshot.get() == INVALID_HANDLE_VALUE) return children;
            PROCESSENTRY32W entry{};
            entry.dwSize = sizeof(entry);
            for (BOOL ok = Process32FirstW(snapshot.get(), &entry); ok; ok = Process32NextW(snapshot.get(), &entry))
            {
                if (entry.th32ProcessID != 0) children.emplace(entry.th32ParentProcessID, entry.th32ProcessID);
            }
            return children;
        }

        std::vector<DWORD> Descend(std::vector<DWORD> const& roots, std::multimap<DWORD, DWORD> const& tree)
        {
            std::vector<DWORD> result;
            std::set<DWORD> seen;
            std::vector<DWORD> queue(roots.begin(), roots.end());
            // this loop walks the tree breadth-first from the port owners to every descendant
            while (!queue.empty())
            {
                DWORD pid = queue.back();
                queue.pop_back();
                if (pid == 0 || !seen.insert(pid).second) continue;
                result.push_back(pid);
                auto [first, last] = tree.equal_range(pid);
                for (auto it = first; it != last; ++it) queue.push_back(it->second);
            }
            return result;
        }

        // ---- launching -------------------------------------------------------------------------
        // Windows command-line quoting (the rules CommandLineToArgvW and the C runtime both parse)
        std::wstring QuoteArg(std::wstring const& arg)
        {
            if (!arg.empty() && arg.find_first_of(L" \t\n\v\"") == std::wstring::npos) return arg;
            std::wstring out = L"\"";
            for (auto it = arg.begin();; ++it)
            {
                size_t backslashes = 0;
                while (it != arg.end() && *it == L'\\')
                {
                    ++it;
                    ++backslashes;
                }
                if (it == arg.end())
                {
                    out.append(backslashes * 2, L'\\');
                    break;
                }
                if (*it == L'"')
                {
                    out.append(backslashes * 2 + 1, L'\\');
                    out.push_back(L'"');
                }
                else
                {
                    out.append(backslashes, L'\\');
                    out.push_back(*it);
                }
            }
            out.push_back(L'"');
            return out;
        }

        // the app's environment plus the engine's extra variables, as a sorted Unicode block
        std::wstring BuildEnvironment(std::vector<std::pair<std::wstring, std::wstring>> const& extra)
        {
            struct NoCase
            {
                bool operator()(std::wstring const& a, std::wstring const& b) const { return _wcsicmp(a.c_str(), b.c_str()) < 0; }
            };
            std::map<std::wstring, std::wstring, NoCase> vars;
            if (auto block = GetEnvironmentStringsW())
            {
                for (auto entry = block; *entry; entry += wcslen(entry) + 1)
                {
                    std::wstring line(entry);
                    auto equals = line.find(L'=', 1);   // per-drive entries such as "=C:=C:\x" start with '='
                    if (equals != std::wstring::npos) vars[line.substr(0, equals)] = line.substr(equals + 1);
                }
                FreeEnvironmentStringsW(block);
            }
            for (auto const& [name, value] : extra) vars[name] = paths::Expand(value);
            std::wstring out;
            for (auto const& [name, value] : vars)
            {
                out += name;
                out += L'=';
                out += value;
                out.push_back(L'\0');
            }
            out.push_back(L'\0');
            return out;
        }

        std::wstring LastErrorText(DWORD code = GetLastError())
        {
            LPWSTR buffer = nullptr;
            FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code, 0,
                           reinterpret_cast<LPWSTR>(&buffer), 0, nullptr);
            std::wstring text = buffer ? buffer : L"error " + std::to_wstring(code);
            LocalFree(buffer);
            while (!text.empty() && (text.back() == L'\n' || text.back() == L'\r' || text.back() == L' ')) text.pop_back();
            return text;
        }

        wil::unique_handle NewKillOnCloseJob()
        {
            wil::unique_handle job(CreateJobObjectW(nullptr, nullptr));
            if (job)
            {
                JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
                limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                SetInformationJobObject(job.get(), JobObjectExtendedLimitInformation, &limits, sizeof(limits));
            }
            return job;
        }

        // Starts one process suspended, puts it in the job, then lets it run, so even its first
        // child process is inside the job. No console window; stdin is NUL; stdout and stderr
        // append to the engine log. Only those two handles are inherited, nothing else of ours.
        wil::unique_handle Launch(EngineProcessSpec const& spec, HANDLE job, HANDLE logFile, std::wstring& error)
        {
            auto exe = paths::Expand(spec.exe);
            std::wstring commandLine = QuoteArg(exe);
            for (auto const& arg : spec.args)
            {
                commandLine += L' ';
                commandLine += QuoteArg(paths::Expand(arg));
            }
            auto cwd = paths::Expand(spec.cwd);
            auto environment = BuildEnvironment(spec.env);

            SECURITY_ATTRIBUTES inheritable{ sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE };
            wil::unique_hfile nul(CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &inheritable, OPEN_EXISTING, 0, nullptr));
            HANDLE inherited[2] = { nul.get(), logFile };

            SIZE_T attributeSize = 0;
            InitializeProcThreadAttributeList(nullptr, 1, 0, &attributeSize);
            std::vector<uint8_t> attributeBuffer(attributeSize);
            auto attributes = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributeBuffer.data());
            if (!InitializeProcThreadAttributeList(attributes, 1, 0, &attributeSize) ||
                !UpdateProcThreadAttribute(attributes, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited, sizeof(inherited), nullptr, nullptr))
            {
                error = L"Could not prepare the process: " + LastErrorText();
                return {};
            }

            STARTUPINFOEXW startup{};
            startup.StartupInfo.cb = sizeof(startup);
            startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
            startup.StartupInfo.hStdInput = nul.get();
            startup.StartupInfo.hStdOutput = logFile;
            startup.StartupInfo.hStdError = logFile;
            startup.lpAttributeList = attributes;

            PROCESS_INFORMATION info{};
            BOOL created = CreateProcessW(nullptr, commandLine.data(), nullptr, nullptr, TRUE,
                                          CREATE_SUSPENDED | CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT,
                                          environment.data(), cwd.empty() ? nullptr : cwd.c_str(), &startup.StartupInfo, &info);
            DWORD createError = GetLastError();
            DeleteProcThreadAttributeList(attributes);
            if (!created)
            {
                error = L"Could not start " + exe + L": " + LastErrorText(createError);
                return {};
            }
            wil::unique_handle process(info.hProcess);
            wil::unique_handle thread(info.hThread);
            if (!AssignProcessToJobObject(job, process.get()))
            {
                error = L"Could not place " + exe + L" under AIWF Studio's control: " + LastErrorText();
                TerminateProcess(process.get(), 1);
                return {};
            }
            ResumeThread(thread.get());
            return process;
        }

        // ---- manifest parsing ------------------------------------------------------------------
        uint32_t ParseColor(std::wstring const& text, uint32_t fallback)
        {
            if (text.size() != 7 || text[0] != L'#') return fallback;
            return 0xFF000000u | static_cast<uint32_t>(std::wcstoul(text.c_str() + 1, nullptr, 16));
        }

        EngineProcessSpec ParseProcess(JsonObject const& obj)
        {
            EngineProcessSpec spec;
            spec.exe = http::Str(obj, L"exe");
            for (auto const& value : http::Arr(obj, L"args"))
            {
                if (value.ValueType() == JsonValueType::String) spec.args.emplace_back(value.GetString());
            }
            spec.cwd = http::Str(obj, L"cwd");
            if (auto env = http::Obj(obj, L"env"))
            {
                for (auto const& pair : env)
                {
                    if (pair.Value().ValueType() == JsonValueType::String) spec.env.emplace_back(pair.Key(), pair.Value().GetString());
                }
            }
            spec.readyUrl = http::Str(obj, L"ready_url");
            spec.readyTimeoutSeconds = static_cast<uint32_t>(http::Num(obj, L"ready_timeout_seconds", 60));
            return spec;
        }
    }

    // ---- public helpers ----------------------------------------------------------------------------
    std::vector<DWORD> ListeningPids(std::vector<uint16_t> const& ports)
    {
        std::vector<DWORD> pids;
        DWORD size = 0;
        GetExtendedTcpTable(nullptr, &size, FALSE, AF_INET, TCP_TABLE_OWNER_PID_LISTENER, 0);
        std::vector<uint8_t> buffer(size + 4096);
        size = static_cast<DWORD>(buffer.size());
        if (GetExtendedTcpTable(buffer.data(), &size, FALSE, AF_INET, TCP_TABLE_OWNER_PID_LISTENER, 0) != NO_ERROR) return pids;
        auto table = reinterpret_cast<MIB_TCPTABLE_OWNER_PID const*>(buffer.data());
        // this loop keeps the owners of the engine's listening ports (port numbers are big-endian)
        for (DWORD i = 0; i < table->dwNumEntries; ++i)
        {
            auto raw = table->table[i].dwLocalPort;
            auto port = static_cast<uint16_t>(((raw & 0xFF) << 8) | ((raw >> 8) & 0xFF));
            if (std::find(ports.begin(), ports.end(), port) != ports.end() &&
                std::find(pids.begin(), pids.end(), table->table[i].dwOwningPid) == pids.end())
            {
                pids.push_back(table->table[i].dwOwningPid);
            }
        }
        return pids;
    }

    std::vector<DWORD> WithDescendants(std::vector<DWORD> const& roots)
    {
        return Descend(roots, ProcessTree());
    }

    std::wstring StateLabel(EngineState state)
    {
        switch (state)
        {
        case EngineState::Stopped: return L"Off";
        case EngineState::Starting: return L"Starting";
        case EngineState::Running: return L"Running";
        case EngineState::RunningExternal: return L"Running (external)";
        case EngineState::Failed: return L"Problem";
        default: return L"Checking";
        }
    }

    // ---- loading the manifest ----------------------------------------------------------------------
    EngineSupervisor& EngineSupervisor::Instance()
    {
        static EngineSupervisor* instance = new EngineSupervisor();   // lives for the whole process
        return *instance;
    }

    bool EngineSupervisor::Load(std::wstring& error)
    {
        auto userManifest = paths::DataDir() / L"engines.json";
        auto shippedManifest = paths::ExeDir() / L"engines.json";
        auto path = std::filesystem::exists(userManifest) ? userManifest : shippedManifest;
        auto text = paths::ReadUtf8(path);
        if (!text)
        {
            error = L"The engine list was not found: " + path.wstring();
            return false;
        }
        JsonObject root{ nullptr };
        if (!JsonObject::TryParse(to_hstring(*text), root))
        {
            error = L"The engine list is not valid JSON: " + path.wstring();
            return false;
        }

        std::vector<EngineSpec> engines;
        // this loop reads each engine; entries without an id or health URL are skipped
        for (auto const& value : http::Arr(root, L"engines"))
        {
            if (value.ValueType() != JsonValueType::Object) continue;
            auto obj = value.GetObject();
            EngineSpec spec;
            spec.id = http::Str(obj, L"id");
            spec.name = http::Str(obj, L"name", spec.id.c_str());
            spec.role = http::Str(obj, L"role");
            spec.healthUrl = http::Str(obj, L"health_url");
            spec.color = ParseColor(std::wstring(http::Str(obj, L"color")), spec.color);
            spec.usesGpu = http::Bool(obj, L"uses_gpu");
            for (auto const& port : http::Arr(obj, L"ports"))
            {
                if (port.ValueType() == JsonValueType::Number) spec.ports.push_back(static_cast<uint16_t>(port.GetNumber()));
            }
            for (auto const& process : http::Arr(obj, L"processes"))
            {
                if (process.ValueType() == JsonValueType::Object) spec.processes.push_back(ParseProcess(process.GetObject()));
            }
            if (!spec.id.empty() && !spec.healthUrl.empty()) engines.push_back(std::move(spec));
        }

        std::map<std::wstring, std::wstring> endpoints;
        if (auto block = http::Obj(root, L"endpoints"))
        {
            for (auto const& pair : block)
            {
                if (pair.Value().ValueType() == JsonValueType::String) endpoints[std::wstring(pair.Key())] = paths::Expand(std::wstring(pair.Value().GetString()));
            }
        }

        std::lock_guard guard(m_lock);
        m_engines = std::move(engines);
        m_endpoints = std::move(endpoints);
        m_manifestPath = path;
        for (auto const& engine : m_engines) m_status.try_emplace(engine.id);
        return true;
    }

    std::filesystem::path EngineSupervisor::ManifestPath() const
    {
        std::lock_guard guard(m_lock);
        return m_manifestPath;
    }

    std::vector<EngineSpec> EngineSupervisor::Engines() const
    {
        std::lock_guard guard(m_lock);
        return m_engines;
    }

    EngineSpec const* EngineSupervisor::Find(std::wstring const& id) const
    {
        std::lock_guard guard(m_lock);
        for (auto const& engine : m_engines)
        {
            if (engine.id == id) return &engine;
        }
        return nullptr;
    }

    std::wstring EngineSupervisor::Endpoint(std::wstring const& key, std::wstring const& fallback) const
    {
        std::lock_guard guard(m_lock);
        auto it = m_endpoints.find(key);
        return it != m_endpoints.end() && !it->second.empty() ? it->second : fallback;
    }

    EngineStatus EngineSupervisor::Status(std::wstring const& id) const
    {
        std::lock_guard guard(m_lock);
        auto it = m_status.find(id);
        return it != m_status.end() ? it->second : EngineStatus{};
    }

    bool EngineSupervisor::IsAnswering(std::wstring const& id) const
    {
        auto state = Status(id).state;
        return state == EngineState::Running || state == EngineState::RunningExternal;
    }

    // ---- status bookkeeping ---------------------------------------------------------------------------
    uint64_t EngineSupervisor::StopGeneration() const
    {
        std::lock_guard guard(m_lock);
        return m_stopGeneration;
    }

    bool EngineSupervisor::HasPendingStarts() const
    {
        std::lock_guard guard(m_lock);
        return !m_starting.empty();
    }

    std::wstring EngineSupervisor::HealthBearer(std::wstring const& id) const
    {
        if (id == L"chat") return std::wstring(AppState::Get().ChatKey());
        if (id != L"datasets") return {};
        return paths::FirstKeyLine(DatasetTokenFile());
    }

    std::wstring EngineSupervisor::DatasetTokenFile() const
    {
        std::wstring configuredState, configuredToken;
        std::vector<std::pair<std::wstring, std::wstring>> effective;
        for (auto const& engine : Engines())
        {
            if (engine.id == L"datasets")
                for (auto const& process : engine.processes) effective.insert(effective.end(), process.env.begin(), process.env.end());
        }
        auto saved = AppState::Get().EngineEnv(L"datasets");
        effective.insert(effective.end(), saved.begin(), saved.end());
        for (auto const& [key, value] : effective)
        {
            if (_wcsicmp(key.c_str(), L"DATASET_STUDIO_STATE") == 0) configuredState = paths::Expand(value);
            if (_wcsicmp(key.c_str(), L"AIWF_DATASET_STUDIO_TOKEN_FILE") == 0) configuredToken = paths::Expand(value);
        }
        auto file = Endpoint(L"dataset_key_file", L"");
        if (file.empty()) file = configuredToken;
        if (file.empty() && !configuredState.empty()) file = (std::filesystem::path(configuredState) / L"api-token.txt").wstring();
        if (file.empty()) file = paths::Expand(L"%AIWF_DATASET_STUDIO_TOKEN_FILE%");
        if (file.empty() || file.find(L'%') != std::wstring::npos)
        {
            auto state = paths::Expand(L"%DATASET_STUDIO_STATE%");
            if (state.empty() || state.find(L'%') != std::wstring::npos) state = L"F:\\Dataset Studio";
            file = (std::filesystem::path(state) / L"api-token.txt").wstring();
        }
        return file;
    }

    void EngineSupervisor::SetStatus(std::wstring const& id, EngineState state, std::wstring detail, uint64_t generation, uint64_t startToken)
    {
        std::lock_guard guard(m_lock);
        auto pending = m_starting.find(id);
        if (generation != m_stopGeneration || AppState::Get().ShuttingDown() || pending == m_starting.end() || pending->second != startToken) return;
        auto& status = m_status[id];
        status.state = state;
        status.detail = std::move(detail);
    }

    std::vector<DWORD> EngineSupervisor::OwnedPids(std::wstring const& id) const
    {
        std::shared_ptr<Owned> owned;
        {
            std::lock_guard guard(m_lock);
            auto it = m_owned.find(id);
            if (it != m_owned.end()) owned = it->second;
        }
        return owned && owned->job ? JobPids(owned->job.get()) : std::vector<DWORD>{};
    }

    event_token EngineSupervisor::StatusChanged(std::function<void()> handler)
    {
        std::lock_guard guard(m_lock);
        event_token token{ m_nextToken++ };
        m_handlers[token.value] = std::move(handler);
        return token;
    }

    void EngineSupervisor::StatusChanged(event_token token)
    {
        std::lock_guard guard(m_lock);
        m_handlers.erase(token.value);
    }

    void EngineSupervisor::ClearHandlers()
    {
        std::lock_guard guard(m_lock);
        m_handlers.clear();
    }

    void EngineSupervisor::RaiseChanged()
    {
        std::vector<std::function<void()>> handlers;
        {
            std::lock_guard guard(m_lock);
            for (auto const& [token, handler] : m_handlers) handlers.push_back(handler);
        }
        for (auto const& handler : handlers) handler();
    }

    // ---- probing ----------------------------------------------------------------------------------------
    IAsyncAction EngineSupervisor::RefreshAsync()
    {
        co_await resume_background();
        auto generation = StopGeneration();
        auto engines = Engines();
        std::map<std::wstring, uint64_t> probeGenerations;
        {
            std::lock_guard guard(m_lock);
            probeGenerations = m_engineGeneration;
        }

        // this block probes every engine at once, then waits for all answers
        std::vector<IAsyncOperation<bool>> probes;
        for (auto const& engine : engines) probes.push_back(http::ReadyAsync(hstring(engine.healthUrl), 1500, hstring(HealthBearer(engine.id))));
        auto tree = ProcessTree();

        for (size_t i = 0; i < engines.size(); ++i)
        {
            auto const& engine = engines[i];
            bool answers = co_await probes[i];
            auto ownPids = OwnedPids(engine.id);
            bool ownAlive = !ownPids.empty();
            std::vector<DWORD> pids = ownAlive ? ownPids : (answers ? Descend(ListeningPids(engine.ports), tree) : std::vector<DWORD>{});

            std::lock_guard guard(m_lock);
            if (generation != m_stopGeneration || AppState::Get().ShuttingDown()) co_return;
            if (probeGenerations[engine.id] != m_engineGeneration[engine.id] || m_starting.count(engine.id)) continue;   // StartAsync reports progress itself
            auto& status = m_status[engine.id];
            status.pids = std::move(pids);
            if (answers)
            {
                status.state = ownAlive ? EngineState::Running : EngineState::RunningExternal;
                status.detail = ownAlive ? L"Started by AIWF Studio." : L"Running; started outside AIWF Studio, so it is left as is.";
            }
            else if (ownAlive)
            {
                status.state = EngineState::Starting;
                status.detail = L"Running but not answering yet (loading or busy).";
            }
            else if (status.state == EngineState::Running || status.state == EngineState::Starting)
            {
                status.state = EngineState::Failed;
                status.detail = L"Stopped unexpectedly. Open its log in Settings to see why.";
                m_owned.erase(engine.id);
            }
            else if (status.state != EngineState::Failed)
            {
                status.state = EngineState::Stopped;
                status.detail = L"Not running.";
            }
        }
        RaiseChanged();
    }

    // ---- starting -----------------------------------------------------------------------------------------
    IAsyncAction EngineSupervisor::StartAsync(std::wstring id, std::optional<uint64_t> expectedGeneration)
    {
        auto generation = expectedGeneration.value_or(StopGeneration());
        EngineSpec spec;
        uint64_t startToken = 0;
        {
            std::lock_guard guard(m_lock);
            auto found = std::find_if(m_engines.begin(), m_engines.end(), [&](EngineSpec const& engine) { return engine.id == id; });
            if (generation != m_stopGeneration || AppState::Get().ShuttingDown() || found == m_engines.end() || m_starting.count(id)) co_return;
            spec = *found;
            startToken = ++m_nextStartToken;
            m_starting[id] = startToken;
            ++m_engineGeneration[id];
        }
        // whatever happens below, this engine leaves the "starting" set when the coroutine ends
        auto leaveStarting = wil::scope_exit([this, id, generation, startToken]
        {
            std::lock_guard guard(m_lock);
            auto pending = m_starting.find(id);
            if (generation == m_stopGeneration && pending != m_starting.end() && pending->second == startToken) m_starting.erase(pending);
        });

        SetStatus(id, EngineState::Starting, L"Checking engine...", generation, startToken);
        RaiseChanged();
        co_await resume_background();
        {
            std::lock_guard guard(m_lock);
            auto pending = m_starting.find(id);
            if (generation != m_stopGeneration || AppState::Get().ShuttingDown() || pending == m_starting.end() || pending->second != startToken) co_return;
        }

        if (co_await http::AnswersAsync(hstring(spec.healthUrl), 1500))
        {
            bool ready = co_await http::ReadyAsync(hstring(spec.healthUrl), 1500, hstring(HealthBearer(id)));
            SetStatus(id, ready ? (OwnedPids(id).empty() ? EngineState::RunningExternal : EngineState::Running) : EngineState::Failed,
                      ready ? L"Already running." : L"A service is listening, but its health check failed. Check its log and credentials.", generation, startToken);
            RaiseChanged();
            co_return;
        }
        if (spec.processes.empty())
        {
            SetStatus(id, EngineState::Failed, L"engines.json does not say how to start this engine.", generation, startToken);
            RaiseChanged();
            co_return;
        }

        SetStatus(id, EngineState::Starting, L"Starting...", generation, startToken);
        RaiseChanged();

        auto owned = std::make_shared<Owned>();
        owned->job = NewKillOnCloseJob();
        owned->logPath = (paths::LogsDir() / (id + L"-" + paths::Timestamp() + L".log")).wstring();
        SECURITY_ATTRIBUTES inheritable{ sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE };
        wil::unique_hfile log(CreateFileW(owned->logPath.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                          &inheritable, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr));
        if (!owned->job || !log)
        {
            SetStatus(id, EngineState::Failed, L"Could not create the engine's job or log file: " + LastErrorText(), generation, startToken);
            RaiseChanged();
            co_return;
        }
        {
            std::lock_guard guard(m_lock);
            if (generation != m_stopGeneration || !m_starting.count(id) || m_starting.at(id) != startToken || AppState::Get().ShuttingDown()) co_return;
            m_owned[id] = owned;
            m_status[id].logPath = owned->logPath;
        }

        // folders chosen in setup (Dataset Studio's catalog, ReTrain's model folders, ...) are added on
        // top of each process's own "env"; later entries win, so the setup choice overrides engines.json
        auto setupEnv = AppState::Get().EngineEnv(id);
        if (id == L"engine-api")
        {
            for (auto const& entry : AppState::Get().EngineEnv(L"datasets"))
            {
                if (_wcsicmp(entry.first.c_str(), L"DATASET_STUDIO_STATE") == 0 ||
                    _wcsicmp(entry.first.c_str(), L"AIWF_DATASET_STUDIO_TOKEN_FILE") == 0) setupEnv.push_back(entry);
            }
        }
        if (id == L"engine-api" || id == L"datasets") setupEnv.emplace_back(L"AIWF_DATASET_STUDIO_TOKEN_FILE", DatasetTokenFile());
        for (auto& process : spec.processes)
        {
            process.env.insert(process.env.end(), setupEnv.begin(), setupEnv.end());
        }

        // this loop starts each process in order and waits until it answers before the next one
        for (size_t index = 0; index < spec.processes.size(); ++index)
        {
            auto const& process = spec.processes[index];
            std::wstring step = spec.processes.size() > 1 ? L" (step " + std::to_wstring(index + 1) + L" of " + std::to_wstring(spec.processes.size()) + L")" : L"";
            if (!process.readyUrl.empty() && co_await http::ReadyAsync(hstring(process.readyUrl), 800, hstring(HealthBearer(id))))
            {
                continue;   // this part is already running (for example the chat backend on its own)
            }
            std::wstring error;
            wil::unique_handle handle;
            HANDLE raw = nullptr;
            {
                std::lock_guard guard(m_lock);
                auto current = m_owned.find(id);
                if (generation != m_stopGeneration || !m_starting.count(id) || m_starting.at(id) != startToken || AppState::Get().ShuttingDown() || current == m_owned.end() || current->second != owned) co_return;
                handle = Launch(process, owned->job.get(), log.get(), error);
                if (handle)
                {
                    raw = handle.get();
                    owned->processes.push_back(std::move(handle));
                }
            }
            if (!raw)
            {
                TerminateJobObject(owned->job.get(), 1);
                SetStatus(id, EngineState::Failed, error, generation, startToken);
                RaiseChanged();
                co_return;
            }
            SetStatus(id, EngineState::Starting, L"Starting" + step + L"...", generation, startToken);
            RaiseChanged();

            auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(process.readyTimeoutSeconds);
            bool ready = process.readyUrl.empty();
            while (!ready)
            {
                if (WaitForSingleObject(raw, 0) == WAIT_OBJECT_0)
                {
                    DWORD code = 0;
                    GetExitCodeProcess(raw, &code);
                    TerminateJobObject(owned->job.get(), 1);
                    SetStatus(id, EngineState::Failed, L"Exited while starting (code " + std::to_wstring(code) + L"). Log: " + owned->logPath, generation, startToken);
                    RaiseChanged();
                    co_return;
                }
                if (std::chrono::steady_clock::now() > deadline)
                {
                    TerminateJobObject(owned->job.get(), 1);
                    SetStatus(id, EngineState::Failed, L"Not ready after " + std::to_wstring(process.readyTimeoutSeconds) + L" s, so it was stopped. Log: " + owned->logPath, generation, startToken);
                    RaiseChanged();
                    co_return;
                }
                ready = co_await http::ReadyAsync(hstring(process.readyUrl), 1500, hstring(HealthBearer(id)));
                if (!ready) co_await resume_after(std::chrono::milliseconds(700));
            }
        }

        {
            std::lock_guard guard(m_lock);
            auto current = m_owned.find(id);
            if (generation != m_stopGeneration || !m_starting.count(id) || m_starting.at(id) != startToken || AppState::Get().ShuttingDown() || current == m_owned.end() || current->second != owned) co_return;
            auto& status = m_status[id];
            status.state = EngineState::Running;
            status.detail = L"Started by AIWF Studio.";
            status.pids = JobPids(owned->job.get());
        }
        RaiseChanged();
    }

    // ---- stopping -------------------------------------------------------------------------------------------
    bool EngineSupervisor::Stop(std::wstring const& id, std::wstring& message)
    {
        std::shared_ptr<Owned> owned;
        {
            std::lock_guard guard(m_lock);
            auto it = m_owned.find(id);
            if (it == m_owned.end() && !m_starting.count(id))
            {
                message = L"This engine was not started by AIWF Studio, so it is left running.";
                return false;
            }
            ++m_engineGeneration[id];
            m_starting.erase(id);
            if (it != m_owned.end())
            {
                owned = it->second;
                m_owned.erase(it);
            }
            auto& status = m_status[id];
            status.state = EngineState::Stopped;
            status.detail = L"Stopped by you.";
            status.pids.clear();
        }
        if (owned && owned->job) TerminateJobObject(owned->job.get(), 0);
        RaiseChanged();
        message = L"Stopped.";
        return true;
    }

    void EngineSupervisor::StopAll()
    {
        std::map<std::wstring, std::shared_ptr<Owned>> owned;
        {
            std::lock_guard guard(m_lock);
            ++m_stopGeneration;
            m_starting.clear();
            owned.swap(m_owned);
        }
        for (auto const& [id, engine] : owned)
        {
            if (engine && engine->job) TerminateJobObject(engine->job.get(), 0);
        }
    }
}
