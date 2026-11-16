#pragma once
#include "httpServer.hpp"
#include "globals.hpp"

#include <cctype>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace projects_event {

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
                else {
                    out += c;
                }
            }
        }
        return out;
    }

    static inline std::string readTextFile(const fs::path& p) {
        std::ifstream f(p, std::ios::binary);
        if (!f) return {};
        std::ostringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }

    static inline std::string trim(const std::string& s) {
        size_t b = 0, e = s.size();
        while (b < e && (s[b] == ' ' || s[b] == '\t' ||
            s[b] == '\r' || s[b] == '\n')) ++b;
        while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' ||
            s[e - 1] == '\r' || s[e - 1] == '\n')) --e;
        return s.substr(b, e - b);
    }

    static inline int countImages(const fs::path& imagesDir) {
        std::error_code ec;
        if (!fs::is_directory(imagesDir, ec)) return 0;
        static const std::unordered_set<std::string> exts = {
            ".jpg", ".jpeg", ".png", ".bmp",
            ".webp", ".gif", ".tif", ".tiff"
        };
        int n = 0;
        for (auto& e : fs::directory_iterator(imagesDir, ec)) {
            if (ec) break;
            if (!e.is_regular_file()) continue;
            std::string ext = e.path().extension().string();
            for (auto& ch : ext) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            if (exts.count(ext)) ++n;
        }
        return n;
    }
    static inline int countClasses(const fs::path& projDir) {
        auto p = projDir / "coco.txt";
        std::error_code ec;
        if (!fs::is_regular_file(p, ec) || ec) return 0;

        std::ifstream f(p);
        if (!f) return 0;

        int n = 0;
        std::string line;
        while (std::getline(f, line)) {
            if (!trim(line).empty()) ++n;
        }
        return n;
    }
    static inline bool readStatus(const fs::path& projDir) {
        auto p = projDir / "status.txt";
        std::error_code ec;
        if (!fs::is_regular_file(p, ec)) return false;
        std::string s = trim(readTextFile(p));
        for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s == "true";
    }

    static inline std::string readType(const fs::path& projDir) {
        auto p = projDir / "type.txt";
        std::error_code ec;
        if (!fs::is_regular_file(p, ec)) return "unknown";
        std::string t = trim(readTextFile(p));
        return t.empty() ? "unknown" : t;
    }

    static inline httpHttpsServer::ApiResponse buildProjects(const fs::path& root) {
        httpHttpsServer::ApiResponse r;
        r.contentType = "application/json; charset=utf-8";

        std::error_code rootEc;
        if (!fs::is_directory(root, rootEc) || rootEc) {
            r.body = "[]";
            return r;
        }

        std::string body = "[";
        bool first = true;

        std::error_code iterEc;
        for (auto it = fs::directory_iterator(root, iterEc);
            it != fs::directory_iterator();
            it.increment(iterEc))
        {
            if (iterEc) break;

            std::error_code entryEc;
            if (!it->is_directory(entryEc) || entryEc) continue;

            const fs::path projDir = it->path();
            const std::string name = projDir.filename().string();

            std::error_code modelEc;
            const bool state = readStatus(projDir);
            const bool canDownload = fs::is_regular_file(projDir / "model.pt", modelEc) && !modelEc;
            const int  images = countImages(projDir / "images");
            const int  classes = countClasses(projDir);
            const std::string type = readType(projDir);

            if (!first) body += ",";
            first = false;

            body += "{";
            body += "\"name\":\"" + jsonEscape(name) + "\",";
            body += "\"type\":\"" + jsonEscape(type) + "\",";
            body += "\"state\":" + std::string(state ? "true" : "false") + ",";
            body += "\"canDownLoadModel\":" + std::string(canDownload ? "true" : "false") + ",";
            body += "\"countImages\":" + std::to_string(images) + ",";
            body += "\"countClass\":" + std::to_string(classes);
            body += "}";
        }

        body += "]";
        r.body = std::move(body);
        return r;
    }

    inline httpHttpsServer::ApiResponse
        handle(const std::string& ip,
            const std::string& /*rest*/,
            const http::request<http::string_body>& req)
    {
        const auto now = steady_clock::now();

        {
            std::lock_guard<std::mutex> lk(g_rateMtx);
            RateState& st = g_rate[ip];
            if (st.blockedUntil > now) {
                auto remain = duration_cast<seconds>(st.blockedUntil - now).count();
                if (remain < 1) remain = 1;

                httpHttpsServer::ApiResponse r;
                r.status = http::status::too_many_requests;
                r.contentType = "application/json; charset=utf-8";
                r.body = "{\"pause\":" + std::to_string(remain) + "}";
                return r;
            }
        }

        auto q = httpHttpsServer::parseQuery(std::string(req.target()));
        auto itL = q.find("login");
        auto itP = q.find("password");

        bool ok = false;
        if (itL != q.end() && itP != q.end()) {
            std::lock_guard<std::mutex> lk(g_accountMtx);
            for (const auto& [l, p] : globals::accaunt) {
                if (l == itL->second && p == itP->second) { ok = true; break; }
            }
        }

        if (ok) {
            return buildProjects(kProjectsRoot);
        }

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

                httpHttpsServer::ApiResponse r;
                r.status = http::status::too_many_requests;
                r.contentType = "application/json; charset=utf-8";
                r.body = "{\"pause\":" + std::to_string(remain) + "}";
                return r;
            }
        }

        httpHttpsServer::ApiResponse r;
        r.contentType = "application/json; charset=utf-8";
        r.body = "false";
        return r;
    }

}