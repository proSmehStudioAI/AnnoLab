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
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace upload_image_event {

    namespace fs = std::filesystem;
    namespace json = boost::json;
    using namespace std::chrono;

    constexpr int  kMaxFailedRequestsPerMinute = 10;
    constexpr auto kWindow = seconds(60);
    constexpr auto kBlockDuration = seconds(60);
    constexpr std::size_t kMaxFileSize = 20ull * 1024ull * 1024ull;
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

    struct FormPart {
        std::string name;
        std::string filename;
        std::string contentType;
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

    static inline bool parseMultipart(const std::string& contentType,
        const std::string& body,
        std::vector<FormPart>& out)
    {
        const std::string bKey = "boundary=";
        auto bPos = contentType.find(bKey);
        if (bPos == std::string::npos) return false;

        std::string boundary = contentType.substr(bPos + bKey.size());
        auto semi = boundary.find(';');
        if (semi != std::string::npos) boundary.resize(semi);
        boundary = trimStr(boundary);
        if (!boundary.empty() && boundary.front() == '"') {
            auto end = boundary.find('"', 1);
            if (end != std::string::npos) boundary = boundary.substr(1, end - 1);
        }
        if (boundary.empty()) return false;

        const std::string delim = "--" + boundary;
        const std::string crlfDelim = "\r\n" + delim;
        const std::string lfDelim = "\n" + delim;

        size_t pos = body.find(delim);
        if (pos == std::string::npos) return false;
        pos += delim.size();

        if (pos + 1 < body.size() && body[pos] == '\r' && body[pos + 1] == '\n') pos += 2;
        else if (pos < body.size() && body[pos] == '\n') pos += 1;

        while (pos < body.size()) {
            size_t next = body.find(crlfDelim, pos);
            size_t skip = 2;
            if (next == std::string::npos) {
                next = body.find(lfDelim, pos);
                skip = 1;
            }
            if (next == std::string::npos) return false;

            std::string chunk = body.substr(pos, next - pos);

            size_t hEnd = chunk.find("\r\n\r\n");
            size_t hEndLen = 4;
            if (hEnd == std::string::npos) {
                hEnd = chunk.find("\n\n");
                hEndLen = 2;
            }

            FormPart part;
            if (hEnd != std::string::npos) {
                std::string headers = chunk.substr(0, hEnd);
                part.data = chunk.substr(hEnd + hEndLen);

                size_t hp = 0;
                while (hp < headers.size()) {
                    size_t lineEnd = headers.find("\r\n", hp);
                    size_t lineEndLen = 2;
                    if (lineEnd == std::string::npos) {
                        lineEnd = headers.find("\n", hp);
                        lineEndLen = 1;
                    }
                    if (lineEnd == std::string::npos) lineEnd = headers.size();

                    std::string line = headers.substr(hp, lineEnd - hp);
                    auto colon = line.find(':');
                    if (colon != std::string::npos) {
                        std::string hname = toLower(trimStr(line.substr(0, colon)));
                        std::string hval = trimStr(line.substr(colon + 1));

                        if (hname == "content-disposition") {
                            auto np = hval.find("name=\"");
                            if (np != std::string::npos) {
                                auto ne = hval.find('"', np + 6);
                                if (ne != std::string::npos)
                                    part.name = hval.substr(np + 6, ne - np - 6);
                            }
                            auto fp = hval.find("filename=\"");
                            if (fp != std::string::npos) {
                                auto fe = hval.find('"', fp + 10);
                                if (fe != std::string::npos)
                                    part.filename = hval.substr(fp + 10, fe - fp - 10);
                            }
                        }
                        else if (hname == "content-type") {
                            part.contentType = hval;
                        }
                    }

                    hp = lineEnd + lineEndLen;
                    if (lineEnd == headers.size()) break;
                }
            }

            out.push_back(std::move(part));

            pos = next + skip + delim.size();
            if (pos + 1 < body.size() && body[pos] == '-' && body[pos + 1] == '-') break;
            if (pos + 1 < body.size() && body[pos] == '\r' && body[pos + 1] == '\n') pos += 2;
            else if (pos < body.size() && body[pos] == '\n') pos += 1;
        }

        return !out.empty();
    }

    static inline std::string pathBasename(const std::string& raw) {
        auto slash = raw.find_last_of("/\\");
        std::string name = (slash == std::string::npos) ? raw : raw.substr(slash + 1);
        return name;
    }

    static inline bool hasAllowedExt(const std::string& filename) {
        auto dot = filename.find_last_of('.');
        if (dot == std::string::npos || dot == 0) return false;
        std::string ext = toLower(filename.substr(dot));
        return ext == ".jpg" || ext == ".jpeg" || ext == ".png" ||
            ext == ".bmp" || ext == ".webp";
    }

    static inline std::string resolveCollision(const fs::path& dir,
        const std::string& filename)
    {
        auto dot = filename.find_last_of('.');
        std::string base = (dot == std::string::npos) ? filename : filename.substr(0, dot);
        std::string ext = (dot == std::string::npos) ? "" : filename.substr(dot);

        std::unordered_set<std::string> usedBases;
        std::error_code ec;
        if (fs::is_directory(dir, ec) && !ec) {
            for (auto it = fs::directory_iterator(dir, ec);
                it != fs::directory_iterator(); it.increment(ec)) {
                if (ec) break;
                std::error_code fec;
                if (!it->is_regular_file(fec) || fec) continue;
                usedBases.insert(it->path().stem().string());
            }
        }

        auto available = [&](const std::string& b) {
            return !usedBases.count(b);
            };

        if (available(base) && !fs::exists(dir / (base + ext), ec))
            return base + ext;

        for (int i = 1; i < 100000; ++i) {
            std::string candBase = base + "_" + std::to_string(i);
            if (available(candBase)) return candBase + ext;
        }

        std::mt19937_64 gen{ std::random_device{}() };
        for (int i = 0; i < 1000; ++i) {
            std::string candBase = base + "_" + std::to_string(gen());
            if (available(candBase)) return candBase + ext;
        }
        return base + ext;
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

        auto ctIt = req.find(http::field::content_type);
        if (ctIt == req.end()) {
            return mkError("content-type required");
        }
        std::string contentType(ctIt->value());
        if (toLower(contentType).find("multipart/form-data") == std::string::npos) {
            return mkError("expected multipart/form-data");
        }

        std::vector<FormPart> parts;
        if (!parseMultipart(contentType, std::string(req.body()), parts)) {
            return mkError("invalid multipart body");
        }

        std::string login, password, project;
        FormPart    filePart;
        bool        hasFile = false;

        for (auto& p : parts) {
            if (p.name == "login")    login = p.data;
            else if (p.name == "password") password = p.data;
            else if (p.name == "project")  project = p.data;
            else if (p.name == "file") { filePart = std::move(p); hasFile = true; }
        }

        login = trimStr(login);
        password = trimStr(password);
        project = trimStr(project);

        if (login.empty())    return mkError("login required");
        if (password.empty()) return mkError("password required");
        if (project.empty())  return mkError("project required");
        if (!hasFile)         return mkError("file required");

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
        if (!fs::is_directory(projDir, ec) || ec) {
            return mkError("project not found");
        }

        fs::path imagesDir = projDir / "images";
        if (!fs::is_directory(imagesDir, ec)) {
            fs::create_directories(imagesDir, ec);
            if (ec) return mkError("cannot create images dir");
        }

        std::string safeName = pathBasename(filePart.filename);
        safeName = trimStr(safeName);
        if (safeName.empty()) safeName = "upload";

        if (!hasAllowedExt(safeName)) {
            return mkError("Недопустимый формат файла");
        }

        for (char c : safeName) {
            if (c == '/' || c == '\\' || c == '\0' ||
                c == ':' || c == '*' || c == '?' ||
                c == '"' || c == '<' || c == '>' || c == '|')
                return mkError("invalid filename");
        }

        if (filePart.data.empty()) {
            return mkError("empty file");
        }
        if (filePart.data.size() > kMaxFileSize) {
            return mkError("Файл слишком большой");
        }

        std::string finalName;
        {
            std::lock_guard<std::mutex> lk(g_writeMtx);
            finalName = resolveCollision(imagesDir, safeName);

            fs::path target = imagesDir / finalName;
            std::ofstream out(target, std::ios::binary | std::ios::trunc);
            if (!out) return mkError("cannot write file");
            out.write(filePart.data.data(),
                static_cast<std::streamsize>(filePart.data.size()));
            if (!out) return mkError("cannot write file");
        }

        std::string body = "{\"success\":true,\"filename\":\"" +
            jsonEscape(finalName) + "\"}";
        return mkJson(http::status::ok, std::move(body));
    }

}