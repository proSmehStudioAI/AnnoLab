#pragma once
#include "httpServer.hpp"
#include "globals.hpp"

#include <boost/json.hpp>
#include <boost/system/error_code.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace save_annotations_event {

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

    struct Box {
        long long classIndex = 0;
        double    cx = 0.0;
        double    cy = 0.0;
        double    w = 0.0;
        double    h = 0.0;
    };

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

    static inline bool getNum(const json::object& o, std::string_view k, double& out) {
        auto it = o.find(k);
        if (it == o.end()) return false;
        const auto& v = it->value();
        if (v.is_double()) { out = v.as_double(); return true; }
        if (v.is_int64()) { out = static_cast<double>(v.as_int64());  return true; }
        if (v.is_uint64()) { out = static_cast<double>(v.as_uint64()); return true; }
        return false;
    }

    static inline bool getInt(const json::object& o, std::string_view k, long long& out) {
        auto it = o.find(k);
        if (it == o.end()) return false;
        const auto& v = it->value();
        if (v.is_int64()) { out = v.as_int64();  return true; }
        if (v.is_uint64()) { out = static_cast<long long>(v.as_uint64()); return true; }
        if (v.is_double()) { out = static_cast<long long>(v.as_double());  return true; }
        return false;
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
        if (jec || !v.is_object()) {
            return mkError("invalid json body");
        }
        auto& obj = v.as_object();

        auto getStr = [&](std::string_view k, std::string& dst) -> bool {
            auto it = obj.find(k);
            if (it == obj.end() || !it->value().is_string()) return false;
            dst = std::string(it->value().as_string());
            return true;
            };

        std::string login, password, project, filename;
        if (!getStr("login", login) || login.empty())         return mkError("login required");
        if (!getStr("password", password))                     return mkError("password required");
        if (!getStr("project", project) || project.empty())   return mkError("project required");
        if (!getStr("filename", filename) || filename.empty()) return mkError("filename required");

        auto itA = obj.find("annotations");
        if (itA == obj.end() || !itA->value().is_array()) {
            return mkError("annotations must be an array");
        }

        std::vector<Box> boxes;
        boxes.reserve(itA->value().as_array().size());
        for (auto& item : itA->value().as_array()) {
            if (!item.is_object()) return mkError("annotation item must be object");
            const auto& bo = item.as_object();

            Box b;
            double x = 0, y = 0, w = 0, h = 0;
            if (!getInt(bo, "class_index", b.classIndex)) return mkError("class_index required");
            if (!getNum(bo, "x", x)) return mkError("x required");
            if (!getNum(bo, "y", y)) return mkError("y required");
            if (!getNum(bo, "w", w)) return mkError("w required");
            if (!getNum(bo, "h", h)) return mkError("h required");

            if (b.classIndex < 0) return mkError("class_index must be >= 0");
            if (!(w > 0.0) || !(h > 0.0)) return mkError("w and h must be > 0");

            double cx = x + w / 2.0;
            double cy = y + h / 2.0;

            b.cx = std::clamp(cx, 0.0, 1.0);
            b.cy = std::clamp(cy, 0.0, 1.0);
            b.w = std::clamp(w, 0.0, 1.0);
            b.h = std::clamp(h, 0.0, 1.0);

            boxes.push_back(b);
        }

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
        if (!isSafeName(filename)) return mkError("invalid filename");

        std::error_code ec;
        fs::path projDir = kProjectsRoot / project;
        if (!fs::is_directory(projDir, ec) || ec) {
            return mkError("project not found");
        }

        fs::path labelsDir = projDir / "labels";
        if (!fs::is_directory(labelsDir, ec)) {
            fs::create_directories(labelsDir, ec);
            if (ec) return mkError("cannot create labels dir");
        }

        std::string base = stripExtension(filename);
        if (base.empty()) return mkError("invalid filename");

        fs::path labelFile = labelsDir / (base + ".txt");

        std::string content;
        content.reserve(boxes.size() * 40);
        for (auto& b : boxes) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "%lld %.6f %.6f %.6f %.6f\n",
                b.classIndex, b.cx, b.cy, b.w, b.h);
            content += buf;
        }

        {
            std::lock_guard<std::mutex> lk(g_writeMtx);
            std::ofstream out(labelFile, std::ios::binary | std::ios::trunc);
            if (!out) return mkError("cannot write labels file");
            out.write(content.data(), static_cast<std::streamsize>(content.size()));
            if (!out) return mkError("cannot write labels file");
        }

        return mkJson(http::status::ok, "{\"success\":true}");
    }

}