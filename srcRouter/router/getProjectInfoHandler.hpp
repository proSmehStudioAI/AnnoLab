#pragma once
#include "httpServer.hpp"
#include "globals.hpp"
#include "TrainingProcessManager.hpp"

#include <boost/json.hpp>
#include <boost/system/error_code.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>

namespace get_project_info_event {

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
        if (req.method() != http::verb::get) {
            return mkJson(http::status::method_not_allowed,
                "{\"error\":\"GET required\"}");
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

        auto q = httpHttpsServer::parseQuery(std::string(req.target()));
        auto itL = q.find("login");
        auto itP = q.find("password");
        auto itPr = q.find("project");

        if (itL == q.end() || itP == q.end() || itPr == q.end()) {
            return mkError("login, password and project required");
        }

        bool ok = false;
        {
            std::lock_guard<std::mutex> lk(g_accountMtx);
            for (const auto& [l, pw] : globals::accaunt) {
                if (l == itL->second && pw == itP->second) { ok = true; break; }
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

        const std::string& project = itPr->second;
        if (!isSafeName(project)) return mkError("invalid project name");

        std::error_code ec;
        fs::path projDir = kProjectsRoot / project;
        if (!fs::is_directory(projDir, ec) || ec) {
            return mkError("project not found");
        }

        bool state = project_files::readStatusBool(projDir);

        project_files::TrainSettings settings;
        project_files::readSettings(projDir / "modelSettings.json", settings);

        auto metrics = project_files::readResultsCsv(projDir / "train_temp" / "results.csv");

        auto& mgr = training_manager::Manager::instance();
        auto procState = mgr.getState(project);
        const bool hasModel = fs::is_regular_file(projDir / "model.pt", ec) && !ec;

        std::string trainingState;
        switch (procState) {
        case training_manager::ProcState::Running:
            trainingState = "training"; break;
        case training_manager::ProcState::Failed:
            trainingState = "failed";   break;
        case training_manager::ProcState::Stopped:
            trainingState = hasModel ? "done" : "idle"; break;
        case training_manager::ProcState::Finished:
            trainingState = "done";     break;
        case training_manager::ProcState::NeverStarted:
        default:
            if (!state)                      trainingState = "idle";
            else if (hasModel)               trainingState = "done";
            else if (metrics.lastEpoch > 0)  trainingState = "failed";
            else                             trainingState = "idle";
        }

        json::object root;
        root["state"] = state;
        root["trainingState"] = trainingState;
        root["epoch"] = metrics.lastEpoch;
        root["epochsTotal"] = mgr.epochsTotal();

        json::object params;
        params["useGPU"] = settings.useGPU;
        params["modelSelected"] = settings.modelSelected;
        params["imgsz"] = settings.imgsz;
        params["lr"] = settings.lr;
        params["batchSize"] = settings.batchSize;
        params["worker"] = settings.worker;
        root["params"] = std::move(params);

        json::object mets;
        auto addArr = [&](std::string_view key, const std::vector<double>& v) {
            json::array a;
            a.reserve(v.size());
            for (double x : v) a.emplace_back(x);
            mets[key] = std::move(a);
            };
        addArr("box_loss", metrics.box_loss);
        addArr("cls_loss", metrics.cls_loss);
        addArr("dfl_loss", metrics.dfl_loss);
        addArr("precision", metrics.precision);
        addArr("recall", metrics.recall);
        addArr("map50", metrics.map50);
        addArr("map50_95", metrics.map50_95);
        root["metrics"] = std::move(mets);

        return mkJson(http::status::ok, json::serialize(root));
    }

}