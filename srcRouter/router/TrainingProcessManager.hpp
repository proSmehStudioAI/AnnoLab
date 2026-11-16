#pragma once
#include "httpServer.hpp"

#include <boost/json.hpp>
#include <boost/system/error_code.hpp>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/types.h>
#include <sys/wait.h>
#include <csignal>
#include <unistd.h>
#endif

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace project_files {

    namespace fs = std::filesystem;
    namespace json = boost::json;

    inline std::string trimStr(const std::string& s) {
        size_t b = 0, e = s.size();
        while (b < e && (s[b] == ' ' || s[b] == '\t' ||
            s[b] == '\r' || s[b] == '\n')) ++b;
        while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' ||
            s[e - 1] == '\r' || s[e - 1] == '\n')) --e;
        return s.substr(b, e - b);
    }

    inline bool readStatusBool(const fs::path& projDir) {
        std::ifstream f(projDir / "status.txt", std::ios::binary);
        if (!f) return false;
        std::string s;
        std::getline(f, s);
        return trimStr(s) == "true";
    }

    inline bool atomicWrite(const fs::path& target, const std::string& content) {
        fs::path tmp = target;
        tmp += ".tmp";
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            if (!f) return false;
            f.write(content.data(), static_cast<std::streamsize>(content.size()));
            if (!f) return false;
        }
        std::error_code ec;
        fs::rename(tmp, target, ec);
        if (ec) {
            fs::remove(target, ec);
            fs::rename(tmp, target, ec);
            if (ec) { fs::remove(tmp, ec); return false; }
        }
        return true;
    }

    struct TrainSettings {
        bool        useGPU = false;
        std::string modelSelected = "yolo26n";
        int         imgsz = 640;
        double      lr = 0.0001;
        int         batchSize = 16;
        int         worker = 2;
    };

    inline bool readSettings(const fs::path& file, TrainSettings& out) {
        std::ifstream f(file, std::ios::binary);
        if (!f) return false;
        std::string data((std::istreambuf_iterator<char>(f)),
            std::istreambuf_iterator<char>());
        if (data.empty()) return true;

        boost::system::error_code jec;
        auto v = json::parse(data, jec);
        if (jec || !v.is_object()) return false;
        auto& o = v.as_object();

        if (auto it = o.find("useGPU"); it != o.end() && it->value().is_bool())
            out.useGPU = it->value().as_bool();

        if (auto it = o.find("modelSelected"); it != o.end() && it->value().is_string()) {
            out.modelSelected = std::string(it->value().as_string());
            if (out.modelSelected.rfind("yolov8", 0) == 0)
                out.modelSelected = "yolo26" + out.modelSelected.substr(6);
        }

        auto getInt = [&](std::string_view k, int& dst) {
            auto it = o.find(k);
            if (it == o.end()) return;
            const auto& jv = it->value();
            if (jv.is_int64())  dst = static_cast<int>(jv.as_int64());
            else if (jv.is_uint64()) dst = static_cast<int>(jv.as_uint64());
            else if (jv.is_double()) dst = static_cast<int>(jv.as_double());
            };
        getInt("imgsz", out.imgsz);
        getInt("batchSize", out.batchSize);
        getInt("worker", out.worker);

        if (auto it = o.find("lr"); it != o.end()) {
            const auto& jv = it->value();
            if (jv.is_double())      out.lr = jv.as_double();
            else if (jv.is_int64())  out.lr = static_cast<double>(jv.as_int64());
            else if (jv.is_uint64()) out.lr = static_cast<double>(jv.as_uint64());
        }
        return true;
    }

    inline std::string buildSettingsJson(const TrainSettings& s) {
        char lrBuf[64];
        std::snprintf(lrBuf, sizeof(lrBuf), "%.10g", s.lr);
        std::string j;
        j += "{\n";
        j += "  \"useGPU\": " + std::string(s.useGPU ? "true" : "false") + ",\n";
        j += "  \"modelSelected\": \"" + s.modelSelected + "\",\n";
        j += "  \"imgsz\": " + std::to_string(s.imgsz) + ",\n";
        j += "  \"lr\": " + std::string(lrBuf) + ",\n";
        j += "  \"batchSize\": " + std::to_string(s.batchSize) + ",\n";
        j += "  \"worker\": " + std::to_string(s.worker) + "\n";
        j += "}\n";
        return j;
    }

    struct MetricsData {
        std::vector<double> box_loss, cls_loss, dfl_loss;
        std::vector<double> precision, recall, map50, map50_95;
        int                 lastEpoch = 0;
    };

    inline MetricsData readResultsCsv(const fs::path& csvPath) {
        MetricsData out;
        std::ifstream f(csvPath, std::ios::binary);
        if (!f) return out;

        std::string header;
        if (!std::getline(f, header)) return out;
        if (!header.empty() && header.back() == '\r') header.pop_back();

        std::vector<std::string> cols;
        {
            std::istringstream hs(header);
            std::string c;
            while (std::getline(hs, c, ',')) {
                c = trimStr(c);
                cols.push_back(c);
            }
        }
        auto findCol = [&](const std::string& name) -> int {
            for (size_t i = 0; i < cols.size(); ++i)
                if (cols[i] == name) return static_cast<int>(i);
            return -1;
            };

        int iEpoch = findCol("epoch");
        int iBox = findCol("train/box_loss");
        int iCls = findCol("train/cls_loss");
        int iDfl = findCol("train/dfl_loss");
        int iPrec = findCol("metrics/precision(B)");
        int iRec = findCol("metrics/recall(B)");
        int iMap50 = findCol("metrics/mAP50(B)");
        int iMap5095 = findCol("metrics/mAP50-95(B)");

        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;

            std::vector<std::string> vals;
            std::istringstream ls(line);
            std::string v;
            while (std::getline(ls, v, ',')) vals.push_back(v);

            auto num = [&](int i) -> double {
                if (i < 0 || static_cast<size_t>(i) >= vals.size()) return 0.0;
                try { return std::stod(vals[i]); }
                catch (...) { return 0.0; }
                };

            if (iEpoch >= 0 && static_cast<size_t>(iEpoch) < vals.size()) {
                try { out.lastEpoch = static_cast<int>(std::stod(vals[iEpoch])) + 1; }
                catch (...) {}
            }
            out.box_loss.push_back(num(iBox));
            out.cls_loss.push_back(num(iCls));
            out.dfl_loss.push_back(num(iDfl));
            out.precision.push_back(num(iPrec));
            out.recall.push_back(num(iRec));
            out.map50.push_back(num(iMap50));
            out.map50_95.push_back(num(iMap5095));
        }
        return out;
    }

}

namespace training_manager {

    namespace fs = std::filesystem;

    enum class ProcState { NeverStarted, Running, Finished, Failed, Stopped };

    struct ProcInfo {
#ifdef _WIN32
        HANDLE hProcess = nullptr;
        HANDLE hThread = nullptr;
        HANDLE hJob = nullptr;
#endif
        std::mutex        opMtx;
        std::atomic<bool> running{ false };
        std::atomic<ProcState> state{ ProcState::NeverStarted };
        std::atomic<long> exitCode{ 0 };
    };

    struct InferenceProcess {
#ifdef _WIN32
        HANDLE hProcess = nullptr;
        HANDLE hStdinW = nullptr;
        HANDLE hStdoutR = nullptr;
#endif
        std::string       readBuffer;
        std::mutex        mtx;
        std::atomic<bool> running{ false };
    };

    struct InferenceResult {
        bool        ok = false;
        int         count = 0;
        std::string error;
    };

    class Manager {
    public:
        static Manager& instance() {
            static Manager m;
            return m;
        }

        void setPythonCommand(std::string cmd) { pythonCmd_ = std::move(cmd); }
        void setWorkerScript(std::string path) { workerScript_ = std::move(path); }
        void setEpochsTotal(int n) { epochsTotal_ = n; }
        int  epochsTotal() const { return epochsTotal_; }

        bool start(const std::string& project, const fs::path& projectDir) {
            std::shared_ptr<ProcInfo> info;
            {
                std::lock_guard<std::mutex> lk(mtx_);
                auto it = procs_.find(project);
                if (it != procs_.end() && it->second->running.load())
                    return true;
                info = std::make_shared<ProcInfo>();
                procs_[project] = info;
            }
            std::thread([this, info, projectDir]() {
                std::lock_guard<std::mutex> opLk(info->opMtx);
                if (info->running.load()) return;
                if (!spawn(info.get(), projectDir)) {
                    info->running.store(false);
                    info->state.store(ProcState::Failed);
                }
                }).detach();
            return true;
        }

        bool stop(const std::string& project) {
            std::shared_ptr<ProcInfo> info;
            {
                std::lock_guard<std::mutex> lk(mtx_);
                auto it = procs_.find(project);
                if (it == procs_.end()) return true;
                info = it->second;
            }
            std::thread([this, info]() {
                std::lock_guard<std::mutex> opLk(info->opMtx);
                if (!info->running.load()) return;
                terminate(info.get());
                }).detach();
            return true;
        }

        ProcState getState(const std::string& project) {
            std::lock_guard<std::mutex> lk(mtx_);
            auto it = procs_.find(project);
            if (it == procs_.end()) return ProcState::NeverStarted;
            return it->second->state.load();
        }

        bool isRunning(const std::string& project) {
            std::lock_guard<std::mutex> lk(mtx_);
            auto it = procs_.find(project);
            if (it == procs_.end()) return false;
            return it->second->running.load();
        }

        void poll() {
            std::lock_guard<std::mutex> lk(mtx_);
            for (auto& [name, info] : procs_) {
                std::lock_guard<std::mutex> ilk(info->opMtx);
                if (!info->running.load()) continue;
                if (!isAlive(info.get())) {
                    if (info->state.load() == ProcState::Running) {
                        info->state.store(info->exitCode.load() == 0
                            ? ProcState::Finished
                            : ProcState::Failed);
                    }
                    info->running.store(false);
                    closeHandles(info.get());
                }
            }
        }

        InferenceResult runInference(const std::string& project,
            const fs::path& imagePath,
            int timeoutMs = 30000);

        void shutdownInference(const std::string& project);
        void shutdownAllInference();

    private:
        Manager() = default;
        std::mutex mtx_;
        std::unordered_map<std::string, std::shared_ptr<ProcInfo>> procs_;
        std::string pythonCmd_ = "py";
        std::string workerScript_ = "pythonWorkerTrain\\pythonWorkerTrainYolov.py";
        int         epochsTotal_ = 1000;

        bool spawn(ProcInfo* info, const fs::path& projectDir);
        bool terminate(ProcInfo* info);
        bool isAlive(ProcInfo* info);
        void closeHandles(ProcInfo* info);

        std::unordered_map<std::string, std::shared_ptr<InferenceProcess>> infProcs_;
        std::mutex infMtx_;

        std::shared_ptr<InferenceProcess> getOrStartInference(const std::string& project);
        void terminateInference(InferenceProcess* p);
    };

#ifdef _WIN32

    inline bool readLineWithTimeout(HANDLE hPipe, std::string& readBuf,
        std::string& line, int timeoutMs) {
        auto start = std::chrono::steady_clock::now();

        auto nl0 = readBuf.find('\n');
        if (nl0 != std::string::npos) {
            line = readBuf.substr(0, nl0);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            readBuf.erase(0, nl0 + 1);
            return true;
        }

        while (true) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
            if (elapsed >= timeoutMs) return false;

            DWORD avail = 0;
            if (!PeekNamedPipe(hPipe, nullptr, 0, nullptr, &avail, nullptr))
                return false;
            if (avail == 0) {
                Sleep(10);
                continue;
            }

            char buf[512];
            DWORD toRead = avail < sizeof(buf) ? avail : static_cast<DWORD>(sizeof(buf));
            DWORD read = 0;
            if (!ReadFile(hPipe, buf, toRead, &read, nullptr) || read == 0)
                return false;

            readBuf.append(buf, read);

            auto nl = readBuf.find('\n');
            if (nl != std::string::npos) {
                line = readBuf.substr(0, nl);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                readBuf.erase(0, nl + 1);
                return true;
            }
        }
    }

    inline bool Manager::spawn(ProcInfo* info, const fs::path& projectDir) {
        std::error_code ec;
        fs::path workerAbs = fs::absolute(workerScript_, ec).lexically_normal();
        if (ec || !fs::is_regular_file(workerAbs, ec) || ec) {
            std::cerr << "[Manager] worker script not found: "
                << workerAbs.string() << "\n";
            return false;
        }

        fs::path projAbs = fs::absolute(projectDir, ec).lexically_normal();
        if (ec) return false;

        std::string cmd = pythonCmd_ + " -u \"" + workerAbs.string() +
            "\" --projectname \"" +
            projAbs.filename().string() + "\"";

        std::cout << "[Manager] spawn: " << cmd << "\n";

        std::vector<char> buf(cmd.begin(), cmd.end());
        buf.push_back('\0');

        fs::create_directories(projAbs, ec);

        SECURITY_ATTRIBUTES sa{};
        sa.nLength = sizeof(sa);
        sa.bInheritHandle = TRUE;
        sa.lpSecurityDescriptor = nullptr;

        fs::path logPath = projAbs / "worker.log";
        HANDLE hLog = CreateFileA(
            logPath.string().c_str(),
            FILE_APPEND_DATA,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            &sa, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);

        if (hLog == INVALID_HANDLE_VALUE) {
            std::cerr << "[Manager] cannot open worker.log\n";
            return false;
        }

        SIZE_T attrSize = 0;
        InitializeProcThreadAttributeList(nullptr, 1, 0, &attrSize);
        std::vector<BYTE> attrBuf(attrSize);
        auto attrs = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attrBuf.data());

        if (!InitializeProcThreadAttributeList(attrs, 1, 0, &attrSize)) {
            CloseHandle(hLog);
            return false;
        }

        HANDLE inheritHandles[] = { hLog };
        if (!UpdateProcThreadAttribute(
            attrs, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
            inheritHandles, sizeof(inheritHandles), nullptr, nullptr)) {
            DeleteProcThreadAttributeList(attrs);
            CloseHandle(hLog);
            return false;
        }

        STARTUPINFOEXA siex{};
        siex.StartupInfo.cb = sizeof(siex);
        siex.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
        siex.StartupInfo.hStdOutput = hLog;
        siex.StartupInfo.hStdError = hLog;
        siex.StartupInfo.hStdInput = nullptr;
        siex.lpAttributeList = attrs;

        PROCESS_INFORMATION pi{};

        BOOL ok = CreateProcessA(
            nullptr, buf.data(),
            nullptr, nullptr,
            TRUE,
            EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW | CREATE_SUSPENDED,
            nullptr, nullptr,
            &siex.StartupInfo, &pi);

        DWORD err = ok ? 0 : GetLastError();

        DeleteProcThreadAttributeList(attrs);
        CloseHandle(hLog);

        if (!ok) {
            std::cerr << "[Manager] CreateProcessA failed, err=" << err
                << ", cmd=" << cmd << "\n";
            return false;
        }

        HANDLE hJob = CreateJobObjectA(nullptr, nullptr);
        if (hJob) {
            if (!AssignProcessToJobObject(hJob, pi.hProcess)) {
                CloseHandle(hJob);
                hJob = nullptr;
            }
            else {
                JOBOBJECT_EXTENDED_LIMIT_INFORMATION jeli{};
                jeli.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                SetInformationJobObject(hJob, JobObjectExtendedLimitInformation,
                    &jeli, sizeof(jeli));
            }
        }

        ResumeThread(pi.hThread);

        info->hProcess = pi.hProcess;
        info->hThread = pi.hThread;
        info->hJob = hJob;
        info->running.store(true);
        info->state.store(ProcState::Running);
        return true;
    }

    inline bool Manager::terminate(ProcInfo* info) {
        bool alive = true;
        if (info->hProcess) {
            DWORD code = 0;
            if (GetExitCodeProcess(info->hProcess, &code) && code != STILL_ACTIVE) {
                alive = false;
                info->exitCode.store(static_cast<long>(code));
            }
        }

        if (alive) {
            if (info->hJob) {
                TerminateJobObject(info->hJob, 1);
            }
            else if (info->hProcess) {
                TerminateProcess(info->hProcess, 1);
            }
            if (info->hProcess) {
                WaitForSingleObject(info->hProcess, 2000);
                DWORD code = 0;
                if (GetExitCodeProcess(info->hProcess, &code))
                    info->exitCode.store(static_cast<long>(code));
            }
            info->state.store(ProcState::Stopped);
        }
        else {
            if (info->state.load() == ProcState::Running) {
                info->state.store(info->exitCode.load() == 0
                    ? ProcState::Finished
                    : ProcState::Failed);
            }
        }

        info->running.store(false);
        closeHandles(info);
        return true;
    }

    inline bool Manager::isAlive(ProcInfo* info) {
        if (!info->hProcess) return false;
        DWORD code = 0;
        if (!GetExitCodeProcess(info->hProcess, &code)) return false;
        if (code == STILL_ACTIVE) return true;
        info->exitCode.store(static_cast<long>(code));
        return false;
    }

    inline void Manager::closeHandles(ProcInfo* info) {
        if (info->hThread) { CloseHandle(info->hThread);  info->hThread = nullptr; }
        if (info->hProcess) { CloseHandle(info->hProcess); info->hProcess = nullptr; }
        if (info->hJob) { CloseHandle(info->hJob);     info->hJob = nullptr; }
    }

    inline std::shared_ptr<InferenceProcess>
        Manager::getOrStartInference(const std::string& project) {
        std::lock_guard<std::mutex> lk(infMtx_);

        auto it = infProcs_.find(project);
        if (it != infProcs_.end()) {
            auto& p = it->second;
            DWORD code = 0;
            if (p->running.load() && p->hProcess &&
                GetExitCodeProcess(p->hProcess, &code) && code == STILL_ACTIVE) {
                return p;
            }
            terminateInference(p.get());
            infProcs_.erase(it);
        }

        std::error_code ec;
        fs::path workerAbs = fs::absolute(workerScript_, ec).lexically_normal();
        if (ec || !fs::is_regular_file(workerAbs, ec) || ec) {
            std::cerr << "[Manager] inf worker not found: " << workerAbs.string() << "\n";
            return nullptr;
        }

        fs::path projAbs = fs::absolute(fs::path("projects") / project, ec).lexically_normal();
        if (ec) return nullptr;
        fs::create_directories(projAbs, ec);

        auto p = std::make_shared<InferenceProcess>();

        SECURITY_ATTRIBUTES sa{};
        sa.nLength = sizeof(sa);
        sa.bInheritHandle = TRUE;

        HANDLE childStdinR = nullptr;
        HANDLE parentStdinW = nullptr;
        if (!CreatePipe(&childStdinR, &parentStdinW, &sa, 0)) {
            std::cerr << "[Manager] CreatePipe(stdin) failed, err="
                << GetLastError() << "\n";
            return nullptr;
        }
        p->hStdinW = parentStdinW;

        HANDLE childStdoutW = nullptr;
        if (!CreatePipe(&p->hStdoutR, &childStdoutW, &sa, 0)) {
            std::cerr << "[Manager] CreatePipe(stdout) failed, err="
                << GetLastError() << "\n";
            CloseHandle(childStdinR);
            CloseHandle(parentStdinW);
            return nullptr;
        }

        SetHandleInformation(p->hStdinW, HANDLE_FLAG_INHERIT, 0);
        SetHandleInformation(p->hStdoutR, HANDLE_FLAG_INHERIT, 0);

        fs::path logPath = projAbs / "autodetect.log";
        HANDLE hLog = CreateFileA(logPath.string().c_str(), FILE_APPEND_DATA,
            FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
            OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);

        std::string cmd = pythonCmd_ + " -u \"" + workerAbs.string() +
            "\" --projectname \"" + project + "\" --serve";

        std::cout << "[Manager] inf spawn: " << cmd << "\n";

        std::vector<char> buf(cmd.begin(), cmd.end());
        buf.push_back('\0');

        SIZE_T attrSize = 0;
        InitializeProcThreadAttributeList(nullptr, 1, 0, &attrSize);
        std::vector<BYTE> attrBuf(attrSize);
        auto attrs = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attrBuf.data());

        if (!InitializeProcThreadAttributeList(attrs, 1, 0, &attrSize)) {
            if (hLog != INVALID_HANDLE_VALUE) CloseHandle(hLog);
            CloseHandle(childStdinR);
            CloseHandle(childStdoutW);
            CloseHandle(p->hStdinW);  p->hStdinW = nullptr;
            CloseHandle(p->hStdoutR); p->hStdoutR = nullptr;
            return nullptr;
        }

        std::vector<HANDLE> inheritList = { childStdinR, childStdoutW };
        if (hLog != INVALID_HANDLE_VALUE) inheritList.push_back(hLog);

        if (!UpdateProcThreadAttribute(
            attrs, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
            inheritList.data(), inheritList.size() * sizeof(HANDLE),
            nullptr, nullptr)) {
            DeleteProcThreadAttributeList(attrs);
            if (hLog != INVALID_HANDLE_VALUE) CloseHandle(hLog);
            CloseHandle(childStdinR);
            CloseHandle(childStdoutW);
            CloseHandle(p->hStdinW);  p->hStdinW = nullptr;
            CloseHandle(p->hStdoutR); p->hStdoutR = nullptr;
            return nullptr;
        }

        STARTUPINFOEXA siex{};
        siex.StartupInfo.cb = sizeof(siex);
        siex.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
        siex.StartupInfo.hStdInput = childStdinR;
        siex.StartupInfo.hStdOutput = childStdoutW;
        siex.StartupInfo.hStdError = (hLog != INVALID_HANDLE_VALUE) ? hLog : childStdoutW;
        siex.lpAttributeList = attrs;

        PROCESS_INFORMATION pi{};
        BOOL ok = CreateProcessA(nullptr, buf.data(), nullptr, nullptr, TRUE,
            CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT, nullptr, nullptr,
            &siex.StartupInfo, &pi);

        DWORD createErr = ok ? 0 : GetLastError();

        DeleteProcThreadAttributeList(attrs);
        CloseHandle(childStdinR);
        CloseHandle(childStdoutW);
        if (hLog != INVALID_HANDLE_VALUE) CloseHandle(hLog);

        if (!ok) {
            std::cerr << "[Manager] inf CreateProcessA failed, err="
                << createErr << "\n";
            CloseHandle(p->hStdinW);  p->hStdinW = nullptr;
            CloseHandle(p->hStdoutR); p->hStdoutR = nullptr;
            return nullptr;
        }

        CloseHandle(pi.hThread);
        p->hProcess = pi.hProcess;
        p->running.store(true);

        auto start = std::chrono::steady_clock::now();
        bool ready = false;
        while (std::chrono::steady_clock::now() - start < std::chrono::seconds(90)) {
            std::string line;
            if (!readLineWithTimeout(p->hStdoutR, p->readBuffer, line, 1000)) {
                DWORD code = 0;
                if (GetExitCodeProcess(p->hProcess, &code) && code != STILL_ACTIVE) {
                    std::cerr << "[Manager] inf worker died before READY, code="
                        << code << "\n";
                    break;
                }
                continue;
            }
            if (line == "READY") { ready = true; break; }
            std::cerr << "[Manager] inf worker: " << line << "\n";
        }

        if (!ready) {
            terminateInference(p.get());
            return nullptr;
        }

        infProcs_[project] = p;
        return p;
    }

    inline void Manager::terminateInference(InferenceProcess* p) {
        if (!p) return;

        if (p->hStdinW) {
            const char* stop = "__stop__\n";
            DWORD written = 0;
            WriteFile(p->hStdinW, stop, 9, &written, nullptr);
            CloseHandle(p->hStdinW);
            p->hStdinW = nullptr;
        }

        if (p->hProcess) {
            if (WaitForSingleObject(p->hProcess, 3000) != WAIT_OBJECT_0)
                TerminateProcess(p->hProcess, 1);
            WaitForSingleObject(p->hProcess, 1000);
            CloseHandle(p->hProcess);
            p->hProcess = nullptr;
        }

        if (p->hStdoutR) {
            CloseHandle(p->hStdoutR);
            p->hStdoutR = nullptr;
        }

        p->running.store(false);
    }

    inline InferenceResult
        Manager::runInference(const std::string& project,
            const fs::path& imagePath,
            int timeoutMs) {
        InferenceResult res;

        auto p = getOrStartInference(project);
        if (!p) {
            res.error = "cannot start inference worker (see autodetect.log)";
            return res;
        }

        std::lock_guard<std::mutex> lk(p->mtx);

        if (!p->hStdinW) {
            res.error = "inference pipe is closed";
            p->running.store(false);
            return res;
        }

        std::string payload = imagePath.string() + "\n";
        DWORD written = 0;
        if (!WriteFile(p->hStdinW, payload.data(),
            static_cast<DWORD>(payload.size()), &written, nullptr)
            || written != payload.size()) {
            DWORD err = GetLastError();
            res.error = "cannot write to inference worker (err="
                + std::to_string(err) + ")";
            p->running.store(false);
            return res;
        }
        FlushFileBuffers(p->hStdinW);

        std::string resp;
        if (!readLineWithTimeout(p->hStdoutR, p->readBuffer, resp, timeoutMs)) {
            DWORD code = 0;
            if (p->hProcess && GetExitCodeProcess(p->hProcess, &code)
                && code != STILL_ACTIVE) {
                res.error = "inference worker died";
                p->running.store(false);
            }
            else {
                res.error = "inference timeout";
                terminateInference(p.get());
            }
            return res;
        }

        if (resp.rfind("OK ", 0) == 0) {
            res.ok = true;
            try { res.count = std::stoi(resp.substr(3)); }
            catch (...) {}
            return res;
        }
        if (resp.rfind("ERR ", 0) == 0) {
            res.error = resp.substr(4);
            return res;
        }

        res.error = "unexpected response: " + resp;
        return res;
    }

    inline void Manager::shutdownInference(const std::string& project) {
        std::shared_ptr<InferenceProcess> p;
        {
            std::lock_guard<std::mutex> lk(infMtx_);
            auto it = infProcs_.find(project);
            if (it == infProcs_.end()) return;
            p = it->second;
            infProcs_.erase(it);
        }
        terminateInference(p.get());
    }

    inline void Manager::shutdownAllInference() {
        std::vector<std::shared_ptr<InferenceProcess>> toKill;
        {
            std::lock_guard<std::mutex> lk(infMtx_);
            for (auto& [k, v] : infProcs_) toKill.push_back(v);
            infProcs_.clear();
        }
        for (auto& p : toKill) terminateInference(p.get());
    }

#else

    inline bool Manager::spawn(ProcInfo*, const fs::path&) { return false; }
    inline bool Manager::terminate(ProcInfo*) { return true; }
    inline bool Manager::isAlive(ProcInfo*) { return false; }
    inline void Manager::closeHandles(ProcInfo*) {}

    inline std::shared_ptr<InferenceProcess>
        Manager::getOrStartInference(const std::string&) { return nullptr; }
    inline void Manager::terminateInference(InferenceProcess*) {}
    inline InferenceResult
        Manager::runInference(const std::string&, const fs::path&, int) {
        InferenceResult r;
        r.error = "inference not supported on this platform";
        return r;
    }
    inline void Manager::shutdownInference(const std::string&) {}
    inline void Manager::shutdownAllInference() {}

#endif

}