#pragma once
#include "httpServer.hpp"
#include "globals.hpp"
#include "TrainingProcessManager.hpp"

#include <boost/json.hpp>
#include <boost/system/error_code.hpp>

#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>

namespace download_model_event {

    namespace fs = std::filesystem;
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

    static inline std::string percentEncode(const std::string& s) {
        static const char* hex = "0123456789ABCDEF";
        std::string out;
        out.reserve(s.size() * 3);
        for (unsigned char c : s) {
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                (c >= '0' && c <= '9') ||
                c == '-' || c == '_' || c == '.' || c == '!' || c == '~') {
                out.push_back(static_cast<char>(c));
            }
            else {
                out.push_back('%');
                out.push_back(hex[c >> 4]);
                out.push_back(hex[c & 0x0f]);
            }
        }
        return out;
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
            return mkError("Проект не найден");
        }

        project_files::TrainSettings settings;
        if (!project_files::readSettings(projDir / "modelSettings.json", settings)) {
            return mkError("Модель ещё не обучена");
        }
        const std::string& model = settings.modelSelected;

        fs::path liveBest = projDir / "train_temp" / "weights" / "best.pt";
        fs::path expBest = projDir / (model + "_best.pt");
        fs::path liveLast = projDir / "train_temp" / "weights" / "last.pt";
        fs::path expLast = projDir / (model + "_last.pt");

        fs::path found;
        std::string suffix;
        std::error_code fEc;

        if (fs::is_regular_file(liveBest, fEc) && !fEc) {
            found = liveBest;
            suffix = "best.pt";
        }
        else if (fs::is_regular_file(expBest, fEc) && !fEc) {
            found = expBest;
            suffix = "best.pt";
        }
        else if (fs::is_regular_file(liveLast, fEc) && !fEc) {
            found = liveLast;
            suffix = "last.pt";
        }
        else if (fs::is_regular_file(expLast, fEc) && !fEc) {
            found = expLast;
            suffix = "last.pt";
        }
        else {
            return mkError("Модель ещё не обучена");
        }

        std::ifstream in(found, std::ios::binary);
        if (!in) {
            return mkError("cannot read model file");
        }
        std::string data((std::istreambuf_iterator<char>(in)),
            std::istreambuf_iterator<char>());

        httpHttpsServer::ApiResponse r;
        r.status = http::status::ok;
        r.contentType = "application/octet-stream";
        r.body = std::move(data);

        std::string fullName = project + "-" + model + "-" + suffix;
        r.contentDisposition =
            "attachment; filename=\"model.pt\"; filename*=UTF-8''" +
            percentEncode(fullName);

        return r;
    }

}