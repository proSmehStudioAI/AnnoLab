#pragma once
#include "httpServer.hpp"
#include "globals.hpp"

#include <array>
#include <cstdint>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace download_dataset_event {

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

    struct ZipEntry {
        std::string name;
        std::string data;
    };

    static inline std::string toLower(std::string s) {
        for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    }

    static inline std::string trimStr(const std::string& s) {
        size_t b = 0, e = s.size();
        while (b < e && (s[b] == ' ' || s[b] == '\t' ||
            s[b] == '\r' || s[b] == '\n')) ++b;
        while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' ||
            s[e - 1] == '\r' || s[e - 1] == '\n')) --e;
        return s.substr(b, e - b);
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

    static inline bool hasAllowedExt(const std::string& filename) {
        auto dot = filename.find_last_of('.');
        if (dot == std::string::npos || dot == 0) return false;
        std::string ext = toLower(filename.substr(dot));
        return ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
            ext == ".bmp" || ext == ".webp" ||
            ext == ".gif" || ext == ".tif" || ext == ".tiff";
    }

    static inline bool endsWith(const std::string& s, const std::string& suf) {
        return s.size() >= suf.size() &&
            s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
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

    static inline void appendLE16(std::string& out, std::uint16_t v) {
        out.push_back(static_cast<char>(v & 0xff));
        out.push_back(static_cast<char>((v >> 8) & 0xff));
    }
    static inline void appendLE32(std::string& out, std::uint32_t v) {
        out.push_back(static_cast<char>(v & 0xff));
        out.push_back(static_cast<char>((v >> 8) & 0xff));
        out.push_back(static_cast<char>((v >> 16) & 0xff));
        out.push_back(static_cast<char>((v >> 24) & 0xff));
    }

    static inline std::uint32_t crc32(const std::string& data) {
        static const std::array<std::uint32_t, 256> table = [] {
            std::array<std::uint32_t, 256> t{};
            for (std::uint32_t i = 0; i < 256; ++i) {
                std::uint32_t c = i;
                for (int k = 0; k < 8; ++k)
                    c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
                t[i] = c;
            }
            return t;
            }();

        std::uint32_t c = 0xFFFFFFFFu;
        for (unsigned char b : data)
            c = table[(c ^ b) & 0xff] ^ (c >> 8);
        return c ^ 0xFFFFFFFFu;
    }

    static inline std::string buildZip(const std::vector<ZipEntry>& entries) {
        struct Cd {
            std::string name;
            std::uint32_t crc = 0;
            std::uint32_t size = 0;
            std::uint32_t offset = 0;
        };

        std::string out;
        out.reserve(4096);
        std::vector<Cd> cds;
        cds.reserve(entries.size());

        for (auto& e : entries) {
            Cd cd;
            cd.name = e.name;
            cd.crc = crc32(e.data);
            cd.size = static_cast<std::uint32_t>(e.data.size());
            cd.offset = static_cast<std::uint32_t>(out.size());

            appendLE32(out, 0x04034b50u);
            appendLE16(out, 20);
            appendLE16(out, 0x0800);
            appendLE16(out, 0);
            appendLE16(out, 0);
            appendLE16(out, 0);
            appendLE32(out, cd.crc);
            appendLE32(out, cd.size);
            appendLE32(out, cd.size);
            appendLE16(out, static_cast<std::uint16_t>(cd.name.size()));
            appendLE16(out, 0);
            out += cd.name;
            out += e.data;

            cds.push_back(std::move(cd));
        }

        std::uint32_t cdOffset = static_cast<std::uint32_t>(out.size());
        for (auto& cd : cds) {
            appendLE32(out, 0x02014b50u);
            appendLE16(out, 20);
            appendLE16(out, 20);
            appendLE16(out, 0x0800);
            appendLE16(out, 0);
            appendLE16(out, 0);
            appendLE16(out, 0);
            appendLE32(out, cd.crc);
            appendLE32(out, cd.size);
            appendLE32(out, cd.size);
            appendLE16(out, static_cast<std::uint16_t>(cd.name.size()));
            appendLE16(out, 0);
            appendLE16(out, 0);
            appendLE16(out, 0);
            appendLE16(out, 0);
            appendLE32(out, 0);
            appendLE32(out, cd.offset);
            out += cd.name;
        }
        std::uint32_t cdSize = static_cast<std::uint32_t>(out.size()) - cdOffset;

        appendLE32(out, 0x06054b50u);
        appendLE16(out, 0);
        appendLE16(out, 0);
        appendLE16(out, static_cast<std::uint16_t>(cds.size()));
        appendLE16(out, static_cast<std::uint16_t>(cds.size()));
        appendLE32(out, cdSize);
        appendLE32(out, cdOffset);
        appendLE16(out, 0);

        return out;
    }

    static inline bool readWholeFile(const fs::path& p, std::string& out) {
        std::ifstream in(p, std::ios::binary);
        if (!in) return false;
        out.assign(std::istreambuf_iterator<char>(in),
            std::istreambuf_iterator<char>());
        return true;
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
        if (!isSafeName(project)) {
            return mkError("invalid project name");
        }

        std::error_code ec;
        fs::path projDir = kProjectsRoot / project;
        if (!fs::is_directory(projDir, ec) || ec) {
            return mkError("Проект не найден");
        }

        fs::path imagesDir = projDir / "images";
        fs::path labelsDir = projDir / "labels";

        std::unordered_map<std::string, std::string> imagesByBase;

        if (fs::is_directory(imagesDir, ec) && !ec) {
            for (auto it = fs::directory_iterator(imagesDir, ec);
                it != fs::directory_iterator(); it.increment(ec)) {
                if (ec) break;
                std::error_code fec;
                if (!it->is_regular_file(fec) || fec) continue;

                std::string fn = it->path().filename().string();
                if (!hasAllowedExt(fn)) continue;

                std::string base = stripExtension(fn);
                if (!imagesByBase.count(base))
                    imagesByBase.emplace(std::move(base), std::move(fn));
            }
        }

        std::vector<ZipEntry> cocoEntries;
        std::vector<ZipEntry> imageEntries;
        std::vector<ZipEntry> labelEntries;

        {
            fs::path cocoFile = projDir / "coco.txt";
            std::string data;
            if (readWholeFile(cocoFile, data) && !data.empty())
                cocoEntries.push_back({ "coco.txt", std::move(data) });
        }

        if (fs::is_directory(labelsDir, ec) && !ec) {
            for (auto it = fs::directory_iterator(labelsDir, ec);
                it != fs::directory_iterator(); it.increment(ec)) {
                if (ec) break;
                std::error_code fec;
                if (!it->is_regular_file(fec) || fec) continue;

                std::string labelName = it->path().filename().string();
                if (!endsWith(toLower(labelName), ".txt")) continue;

                std::string base = stripExtension(labelName);
                auto imgIt = imagesByBase.find(base);
                if (imgIt == imagesByBase.end()) continue;

                fs::path imgPath = imagesDir / imgIt->second;
                std::string imgData;
                if (!readWholeFile(imgPath, imgData)) continue;

                std::string lblData;
                if (!readWholeFile(it->path(), lblData)) continue;

                imageEntries.push_back({ "images/" + imgIt->second, std::move(imgData) });
                labelEntries.push_back({ "labels/" + labelName,   std::move(lblData) });
            }
        }

        if (imageEntries.empty()) {
            return mkError("Нет размеченных изображений");
        }

        std::vector<ZipEntry> all;
        all.reserve(cocoEntries.size() + imageEntries.size() + labelEntries.size());
        for (auto& e : cocoEntries)  all.push_back(std::move(e));
        for (auto& e : imageEntries) all.push_back(std::move(e));
        for (auto& e : labelEntries) all.push_back(std::move(e));

        std::string zip = buildZip(all);

        httpHttpsServer::ApiResponse r;
        r.status = http::status::ok;
        r.contentType = "application/zip";
        r.body = std::move(zip);

        const std::string filename = project + "-dataset.zip";
        r.contentDisposition =
            "attachment; filename=\"dataset.zip\"; filename*=UTF-8''" +
            percentEncode(filename);

        return r;
    }

}