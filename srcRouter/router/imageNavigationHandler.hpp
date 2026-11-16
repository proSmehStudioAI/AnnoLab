#pragma once
#include "httpServer.hpp"
#include "globals.hpp"

#include <boost/json.hpp>
#include <boost/system/error_code.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace image_nav_event {

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
    inline std::mutex                                 g_historyMtx;

    struct UserHistory {
        std::vector<std::string> items;
        long long                pos = -1;
    };

    struct ProjectHistory {
        std::unordered_map<std::string, std::string> assigned;
        std::unordered_map<std::string, UserHistory> history;
    };

    static inline std::mt19937& rng() {
        thread_local std::mt19937 gen{ std::random_device{}() };
        return gen;
    }

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

    static inline bool loadHistory(const fs::path& f, ProjectHistory& out) {
        std::error_code ec;
        if (!fs::is_regular_file(f, ec) || ec) return true;

        std::ifstream in(f, std::ios::binary);
        if (!in) return false;
        std::string data((std::istreambuf_iterator<char>(in)),
            std::istreambuf_iterator<char>());
        if (data.empty()) return true;

        boost::system::error_code jec;
        auto v = json::parse(data, jec);
        if (jec || !v.is_object()) return true;

        auto& obj = v.as_object();

        if (auto it = obj.find("assigned");
            it != obj.end() && it->value().is_object()) {
            for (auto& [k, vv] : it->value().as_object()) {
                if (vv.is_string())
                    out.assigned[std::string(k)] = std::string(vv.as_string());
            }
        }

        if (auto it = obj.find("history");
            it != obj.end() && it->value().is_object()) {
            for (auto& [user, hv] : it->value().as_object()) {
                if (!hv.is_object()) continue;
                UserHistory uh;
                auto& ho = hv.as_object();

                if (auto ii = ho.find("items");
                    ii != ho.end() && ii->value().is_array()) {
                    for (auto& item : ii->value().as_array()) {
                        if (item.is_string())
                            uh.items.push_back(std::string(item.as_string()));
                    }
                }
                if (auto pi = ho.find("pos"); pi != ho.end()) {
                    if (pi->value().is_int64())
                        uh.pos = pi->value().as_int64();
                    else if (pi->value().is_uint64())
                        uh.pos = static_cast<long long>(pi->value().as_uint64());
                }
                out.history[std::string(user)] = std::move(uh);
            }
        }
        return true;
    }

    static inline bool saveHistory(const fs::path& f, const ProjectHistory& h) {
        std::string j;
        j.reserve(4096);
        j += "{\n\"assigned\":{";
        bool first = true;
        for (auto& [k, v] : h.assigned) {
            if (!first) j += ",";
            first = false;
            j += "\"" + jsonEscape(k) + "\":\"" + jsonEscape(v) + "\"";
        }
        j += "},\n\"history\":{";
        first = true;
        for (auto& [user, uh] : h.history) {
            if (!first) j += ",";
            first = false;
            j += "\"" + jsonEscape(user) + "\":{\"items\":[";
            bool fi = true;
            for (auto& it : uh.items) {
                if (!fi) j += ",";
                fi = false;
                j += "\"" + jsonEscape(it) + "\"";
            }
            j += "],\"pos\":" + std::to_string(uh.pos) + "}";
        }
        j += "}\n}\n";

        std::ofstream out(f, std::ios::binary | std::ios::trunc);
        if (!out) return false;
        out.write(j.data(), static_cast<std::streamsize>(j.size()));
        return static_cast<bool>(out);
    }

    static inline std::vector<std::string> listImages(const fs::path& imagesDir) {
        std::vector<std::string> out;
        std::error_code ec;
        if (!fs::is_directory(imagesDir, ec) || ec) return out;

        static const std::unordered_set<std::string> exts = {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp",
            ".gif", ".tif", ".tiff"
        };

        for (auto it = fs::directory_iterator(imagesDir, ec);
            it != fs::directory_iterator(); it.increment(ec)) {
            if (ec) break;
            std::error_code fec;
            if (!it->is_regular_file(fec) || fec) continue;

            std::string ext = it->path().extension().string();
            for (auto& c : ext)
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (exts.count(ext))
                out.push_back(it->path().filename().string());
        }
        std::sort(out.begin(), out.end());
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
        return mkJson(http::status::ok, "{\"error\":\"" + msg + "\"}");
    }

    static inline httpHttpsServer::ApiResponse mkPause(long long remain) {
        return mkJson(http::status::too_many_requests,
            "{\"pause\":" + std::to_string(remain) + "}");
    }

    static inline httpHttpsServer::ApiResponse makeResult(
        const std::string& filename,
        long long          position,
        long long          historySize,
        long long          totalImages,
        long long          assignedCount)
    {
        const bool canPrev = position > 0;
        const bool canNext = (position + 1 < historySize) ||
            (assignedCount < totalImages);

        std::string body = "{";
        body += "\"filename\":\"" + jsonEscape(filename) + "\",";
        body += "\"position\":" + std::to_string(position) + ",";
        body += "\"historySize\":" + std::to_string(historySize) + ",";
        body += "\"totalImages\":" + std::to_string(totalImages) + ",";
        body += "\"assignedCount\":" + std::to_string(assignedCount) + ",";
        body += "\"canPrev\":" + std::string(canPrev ? "true" : "false") + ",";
        body += "\"canNext\":" + std::string(canNext ? "true" : "false");
        body += "}";

        return mkJson(http::status::ok, std::move(body));
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
        auto itD = q.find("dir");

        if (itL == q.end() || itP == q.end() || itPr == q.end()) {
            return mkError("login, password and project required");
        }

        const std::string& login = itL->second;
        const std::string& password = itP->second;
        const std::string& project = itPr->second;
        std::string        dir = (itD != q.end()) ? itD->second : "next";

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

        if (project.empty() || project.find('/') != std::string::npos ||
            project.find('\\') != std::string::npos ||
            project == "." || project == "..") {
            return mkError("invalid project name");
        }

        std::lock_guard<std::mutex> lk(g_historyMtx);

        std::error_code ec;
        fs::path projDir = kProjectsRoot / project;
        if (!fs::is_directory(projDir, ec) || ec) {
            return mkError("project not found");
        }

        auto allImages = listImages(projDir / "images");
        if (allImages.empty()) {
            return mkError("no images in project");
        }

        fs::path histFile = projDir / ".history.json";
        ProjectHistory hist;
        if (!loadHistory(histFile, hist)) {
            return mkError("cannot read history file");
        }

        auto& uh = hist.history[login];

        auto allAssigned = static_cast<long long>(hist.assigned.size());
        auto totalImages = static_cast<long long>(allImages.size());

        auto save = [&]() { saveHistory(histFile, hist); };

        if (dir == "current") {
            if (!uh.items.empty()) {
                if (uh.pos < 0 || uh.pos >= static_cast<long long>(uh.items.size()))
                    uh.pos = static_cast<long long>(uh.items.size()) - 1;
                return makeResult(uh.items[uh.pos], uh.pos,
                    static_cast<long long>(uh.items.size()),
                    totalImages, allAssigned);
            }
            dir = "next";
        }

        if (dir == "prev") {
            if (uh.items.empty()) {
                return mkError("no history for this user");
            }
            if (uh.pos > 0) {
                --uh.pos;
                save();
            }
            if (uh.pos < 0) uh.pos = 0;
            return makeResult(uh.items[uh.pos], uh.pos,
                static_cast<long long>(uh.items.size()),
                totalImages, allAssigned);
        }

        if (dir != "next") {
            return mkError("invalid dir");
        }

        if (uh.pos + 1 < static_cast<long long>(uh.items.size())) {
            ++uh.pos;
            save();
            return makeResult(uh.items[uh.pos], uh.pos,
                static_cast<long long>(uh.items.size()),
                totalImages, allAssigned);
        }

        std::vector<std::string> free;
        free.reserve(allImages.size());
        for (auto& img : allImages) {
            if (!hist.assigned.count(img)) free.push_back(img);
        }

        if (free.empty()) {
            if (!uh.items.empty()) {
                uh.pos = static_cast<long long>(uh.items.size()) - 1;
                return makeResult(uh.items[uh.pos], uh.pos,
                    static_cast<long long>(uh.items.size()),
                    totalImages, allAssigned);
            }
            return mkError("all images assigned");
        }

        std::uniform_int_distribution<size_t> dist(0, free.size() - 1);
        std::string chosen = free[dist(rng())];

        hist.assigned[chosen] = login;
        uh.items.push_back(chosen);
        uh.pos = static_cast<long long>(uh.items.size()) - 1;

        save();

        return makeResult(chosen, uh.pos,
            static_cast<long long>(uh.items.size()),
            totalImages,
            static_cast<long long>(hist.assigned.size()));
    }

    static inline std::optional<fs::path> safeImagePath(
        const fs::path& root, const std::string& project, const std::string& filename)
    {
        if (project.empty() || filename.empty()) return std::nullopt;
        if (project == "." || project == "..") return std::nullopt;
        if (filename == "." || filename == "..") return std::nullopt;
        if (project.find('/') != std::string::npos) return std::nullopt;
        if (project.find('\\') != std::string::npos) return std::nullopt;
        if (filename.find('/') != std::string::npos) return std::nullopt;
        if (filename.find('\\') != std::string::npos) return std::nullopt;

        fs::path p = root / project / "images" / filename;
        std::error_code ec;
        if (!fs::is_regular_file(p, ec) || ec) return std::nullopt;
        return p;
    }

    static inline std::string mimeForImage(const fs::path& p) {
        std::string ext = p.extension().string();
        for (auto& c : ext)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
        if (ext == ".png")  return "image/png";
        if (ext == ".gif")  return "image/gif";
        if (ext == ".bmp")  return "image/bmp";
        if (ext == ".webp") return "image/webp";
        if (ext == ".svg")  return "image/svg+xml";
        if (ext == ".tif" || ext == ".tiff") return "image/tiff";
        return "application/octet-stream";
    }

    inline httpHttpsServer::ApiResponse
        handleRaw(const std::string& ip,
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
        auto itF = q.find("file");

        if (itL == q.end() || itP == q.end() || itPr == q.end() || itF == q.end()) {
            return mkError("login, password, project, file required");
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

        auto pathOpt = safeImagePath(kProjectsRoot, itPr->second, itF->second);
        if (!pathOpt) {
            return mkJson(http::status::not_found, "{\"error\":\"not found\"}");
        }

        std::ifstream in(*pathOpt, std::ios::binary);
        if (!in) {
            return mkJson(http::status::internal_server_error,
                "{\"error\":\"cannot open file\"}");
        }
        std::string data((std::istreambuf_iterator<char>(in)),
            std::istreambuf_iterator<char>());

        httpHttpsServer::ApiResponse r;
        r.status = http::status::ok;
        r.contentType = mimeForImage(*pathOpt);
        r.body = std::move(data);
        return r;
    }

}