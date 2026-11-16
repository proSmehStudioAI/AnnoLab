#pragma once
#include "httpServer.hpp"
#include "globals.hpp"
#include "TrainingProcessManager.hpp"

#include <boost/json.hpp>
#include <boost/system/error_code.hpp>

#include <chrono>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>

namespace set_project_state_event {

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

        auto getStr = [&](std::string_view k, std::string& dst) -> bool {
            auto it = o.find(k);
            if (it == o.end() || !it->value().is_string()) return false;
            dst = std::string(it->value().as_string());
            return true;
            };

        std::string login, password, project;
        if (!getStr("login", login) || login.empty())         return mkError("login required");
        if (!getStr("password", password))                     return mkError("password required");
        if (!getStr("project", project) || project.empty())   return mkError("project required");

        auto itS = o.find("state");
        if (itS == o.end() || !itS->value().is_bool())
            return mkError("state required");
        bool state = itS->value().as_bool();

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

        {
            std::lock_guard<std::mutex> lk(g_writeMtx);
            if (!project_files::atomicWrite(projDir / "status.txt",
                state ? "true" : "false")) {
                return mkError("cannot write status.txt");
            }
        }

        auto& mgr = training_manager::Manager::instance();

        if (state) {
            if (!mgr.start(project, projDir)) {
                std::lock_guard<std::mutex> lk(g_writeMtx);
                project_files::atomicWrite(projDir / "status.txt", "false");
                return mkError("cannot start training process");
            }
        }
        else {
            mgr.stop(project);
        }

        std::string body = "{\"success\":true,\"state\":";
        body += state ? "true" : "false";
        body += "}";
        return mkJson(http::status::ok, std::move(body));
    }

}