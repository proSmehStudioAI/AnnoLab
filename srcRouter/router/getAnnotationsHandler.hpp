#pragma once
#include "httpServer.hpp"
#include "globals.hpp"

#include <boost/json.hpp>
#include <boost/system/error_code.hpp>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

namespace get_annotations_event {

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
                else {
                    out += c;
                }
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

    static inline std::string stripExtension(const std::string& f) {
        auto dot = f.find_last_of('.');
        if (dot == std::string::npos || dot == 0) return f;
        return f.substr(0, dot);
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
        auto itF = q.find("filename");

        if (itL == q.end() || itP == q.end() ||
            itPr == q.end() || itF == q.end()) {
            return mkError("login, password, project and filename required");
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
        const std::string& filename = itF->second;

        if (!isSafeName(project))  return mkError("invalid project name");
        if (!isSafeName(filename)) return mkError("invalid filename");

        std::error_code ec;
        fs::path projDir = kProjectsRoot / project;
        if (!fs::is_directory(projDir, ec) || ec) {
            return mkError("project not found");
        }

        std::string base = stripExtension(filename);
        if (base.empty()) return mkError("invalid filename");

        fs::path labelFile = projDir / "labels" / (base + ".txt");

        std::ifstream in(labelFile, std::ios::binary);
        if (!in) {
            return mkJson(http::status::ok, "[]");
        }

        std::string body = "[";
        bool first = true;
        std::string line;

        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();

            std::istringstream ss(line);
            long long cls = 0;
            double cx = 0.0, cy = 0.0, w = 0.0, h = 0.0;

            if (!(ss >> cls >> cx >> cy >> w >> h)) continue;

            if (!(w > 0.0) || !(h > 0.0)) continue;

            double x = cx - w / 2.0;
            double y = cy - h / 2.0;

            if (x < 0.0) x = 0.0;
            if (y < 0.0) y = 0.0;
            if (w > 1.0) w = 1.0;
            if (h > 1.0) h = 1.0;
            if (x + w > 1.0) w = 1.0 - x;
            if (y + h > 1.0) h = 1.0 - y;

            if (cls < 0) continue;

            if (!first) body += ",";
            first = false;

            char buf[160];
            std::snprintf(buf, sizeof(buf),
                "{\"class_index\":%lld,\"x\":%.6f,\"y\":%.6f,\"w\":%.6f,\"h\":%.6f}",
                cls, x, y, w, h);
            body += buf;
        }

        body += "]";
        return mkJson(http::status::ok, std::move(body));
    }

}