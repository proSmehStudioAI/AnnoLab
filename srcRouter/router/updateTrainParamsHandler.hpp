#pragma once
#include "httpServer.hpp"
#include "globals.hpp"
#include "TrainingProcessManager.hpp"

#include <boost/json.hpp>
#include <boost/system/error_code.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>

namespace update_train_params_event {

    namespace fs = std::filesystem;
    namespace json = boost::json;
    using namespace std::chrono;

    constexpr int  kMaxFailedRequestsPerMinute = 10;
    constexpr auto kWindow = seconds(60);
    constexpr auto kBlockDuration = seconds(60);
    inline    fs::path kProjectsRoot = "projects";

    struct RateState {
        steady_clock::time_point windowStart{};
        int                      count = 0;
        steady_clock::time_point blockedUntil{};
    };

    inline std::mutex                                 g_rateMtx;
    inline std::unordered_map<std::string, RateState> g_rate;
    inline std::mutex                                 g_accountMtx;
    inline std::mutex                                 g_writeMtx;

    static inline std::string jsonEscape(const std::string& s) {
        std::string out;
        out.reserve(s.size() + 8);
        for (char c : s) {
            switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                }
                else out += c;
            }
        }
        return out;
    }

    static inline bool isSafeName(const std::string& n) {
        if (n.empty() || n.size() > 200) return false;
        if (n == "." || n == "..") return false;
        for (char c : n) {
            if (c == '/' || c == '\\' || c == '\0' ||
                c == ':' || c == '*' || c == '?' ||
                c == '"' || c == '<' || c == '>' || c == '|')
                return false;
        }
        return true;
    }

    static inline httpHttpsServer::ApiResponse mkJson(http::status st, std::string body) {
        httpHttpsServer::ApiResponse r;
        r.status = st;
        r.contentType = "application/json; charset=utf-8";
        r.body = std::move(body);
        return r;
    }
    static inline httpHttpsServer::ApiResponse mkError(std::string msg) {
        return mkJson(http::status::ok, "{\"error\":\"" + jsonEscape(msg) + "\"}");
    }
    static inline httpHttpsServer::ApiResponse mkPause(long long remain) {
        return mkJson(http::status::too_many_requests,
            "{\"pause\":" + std::to_string(remain) + "}");
    }

    inline httpHttpsServer::ApiResponse
        handle(const std::string& ip,
            const std::string& /*rest*/,
            const http::request<http::string_body>& req)
    {
        if (req.method() != http::verb::post) {
            return mkJson(http::status::method_not_allowed,
                "{\"error\":\"POST required\"}");
        }

        const auto now = steady_clock::now();

        {
            std::lock_guard<std::mutex> lk(g_rateMtx);
            RateState& st = g_rate[ip];
            if (st.blockedUntil > now) {
                auto remain = duration_cast<seconds>(st.blockedUntil - now).count();
                if (remain < 1) remain = 1;
                return mkPause(remain);
            }
        }

        boost::system::error_code jec;
        auto v = json::parse(std::string(req.body()), jec);
        if (jec || !v.is_object()) return mkError("invalid json body");
        auto& o = v.as_object();

        auto getStrRoot = [&](std::string_view k, std::string& dst) -> bool {
            auto it = o.find(k);
            if (it == o.end() || !it->value().is_string()) return false;
            dst = std::string(it->value().as_string());
            return true;
            };

        std::string login, password, project;
        if (!getStrRoot("login", login) || login.empty())        return mkError("login required");
        if (!getStrRoot("password", password))                    return mkError("password required");
        if (!getStrRoot("project", project) || project.empty())  return mkError("project required");

        json::object        syntheticParams;
        const json::object* paramsObj = nullptr;

        auto itP = o.find("params");
        if (itP != o.end() && itP->value().is_object()) {
            paramsObj = &itP->value().as_object();
        }
        else {
            static const char* kParamKeys[] = {
                "useGPU", "modelSelected", "imgsz", "lr", "batchSize", "worker"
            };
            for (const char* k : kParamKeys) {
                auto it = o.find(k);
                if (it != o.end())
                    syntheticParams[k] = it->value();
            }
            paramsObj = &syntheticParams;
        }

        auto& po = *paramsObj;

        auto getBoolP = [&](std::string_view k, bool& dst) -> bool {
            auto it = po.find(k);
            if (it == po.end() || !it->value().is_bool()) return false;
            dst = it->value().as_bool();
            return true;
            };
        auto getIntP = [&](std::string_view k, int& dst) -> bool {
            auto it = po.find(k);
            if (it == po.end()) return false;
            const auto& jv = it->value();
            if (jv.is_int64()) { dst = static_cast<int>(jv.as_int64());  return true; }
            if (jv.is_uint64()) { dst = static_cast<int>(jv.as_uint64()); return true; }
            if (jv.is_double()) { dst = static_cast<int>(jv.as_double()); return true; }
            return false;
            };
        auto getDoubleP = [&](std::string_view k, double& dst) -> bool {
            auto it = po.find(k);
            if (it == po.end()) return false;
            const auto& jv = it->value();
            if (jv.is_double()) { dst = jv.as_double(); return true; }
            if (jv.is_int64()) { dst = static_cast<double>(jv.as_int64());  return true; }
            if (jv.is_uint64()) { dst = static_cast<double>(jv.as_uint64()); return true; }
            return false;
            };
        auto getStrP = [&](std::string_view k, std::string& dst) -> bool {
            auto it = po.find(k);
            if (it == po.end() || !it->value().is_string()) return false;
            dst = std::string(it->value().as_string());
            return true;
            };

        project_files::TrainSettings s;
        if (!getBoolP("useGPU", s.useGPU))
            return mkError("useGPU required");
        if (!getStrP("modelSelected", s.modelSelected) || s.modelSelected.empty())
            return mkError("modelSelected required");
        if (s.modelSelected != "yolo26n" && s.modelSelected != "yolo26s" &&
            s.modelSelected != "yolo26m" && s.modelSelected != "yolo26l" &&
            s.modelSelected != "yolo26x")
            return mkError("modelSelected invalid");
        if (!getIntP("imgsz", s.imgsz))
            return mkError("imgsz required");
        if (s.imgsz < 32 || s.imgsz > 4096 || (s.imgsz % 32) != 0)
            return mkError("imgsz must be 32..4096 multiple of 32");
        if (!getDoubleP("lr", s.lr))
            return mkError("lr required");
        if (!(s.lr > 0.0 && s.lr <= 1.0))
            return mkError("lr must be in (0;1]");
        if (!getIntP("batchSize", s.batchSize))
            return mkError("batchSize required");
        if (s.batchSize < 1 || s.batchSize > 512)
            return mkError("batchSize слишком большой или маленький");
        if (!getIntP("worker", s.worker))
            return mkError("worker required");
        if (s.worker < 0 || s.worker > 32)
            return mkError("worker must be 0..32");

        bool ok = false;
        {
            std::lock_guard<std::mutex> lk(g_accountMtx);
            for (const auto& [l, pw] : globals::accaunt) {
                if (l == login && pw == password) { ok = true; break; }
            }
        }

        if (!ok) {
            {
                std::lock_guard<std::mutex> lk(g_rateMtx);
                RateState& st = g_rate[ip];
                if (now - st.windowStart >= kWindow) {
                    st.windowStart = now;
                    st.count = 0;
                }
                ++st.count;
                if (st.count > kMaxFailedRequestsPerMinute) {
                    st.blockedUntil = now + kBlockDuration;
                    auto remain = duration_cast<seconds>(st.blockedUntil - now).count();
                    return mkPause(remain);
                }
            }
            return mkJson(http::status::ok, "false");
        }

        if (!isSafeName(project)) return mkError("invalid project name");

        std::error_code ec;
        fs::path projDir = kProjectsRoot / project;
        if (!fs::is_directory(projDir, ec) || ec)
            return mkError("project not found");

        auto& mgr = training_manager::Manager::instance();
        if (mgr.isRunning(project)) {
            project_files::TrainSettings cur;
            project_files::readSettings(projDir / "modelSettings.json", cur);
            if (cur.modelSelected != s.modelSelected)
                return mkError("Модель нельзя менять во время обучения");
        }

        {
            std::lock_guard<std::mutex> lk(g_writeMtx);
            if (!project_files::atomicWrite(projDir / "modelSettings.json",
                project_files::buildSettingsJson(s))) {
                return mkError("cannot write settings");
            }
        }

        json::object resp;
        resp["success"] = true;
        json::object p;
        p["useGPU"] = s.useGPU;
        p["modelSelected"] = s.modelSelected;
        p["imgsz"] = s.imgsz;
        p["lr"] = s.lr;
        p["batchSize"] = s.batchSize;
        p["worker"] = s.worker;
        resp["params"] = std::move(p);

        return mkJson(http::status::ok, json::serialize(resp));
    }

}