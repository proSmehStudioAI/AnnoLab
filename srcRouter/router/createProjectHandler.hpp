#pragma once
#include "httpServer.hpp"
#include "globals.hpp"

#include <boost/json.hpp>
#include <boost/system/error_code.hpp>

#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace create_project_event {

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
    inline std::function<void(const fs::path&)>       g_trainEnqueue;

    struct CreateParams {
        std::string name;
        std::string type;
        bool        state = false;
        bool        useGPU = false;
        std::string modelSelected = "yolov8n";
        long long   imgsz = 640;
        double      lr = 0.0001;
        long long   batchSize = 16;
        long long   worker = 2;
    };

    struct FormPart {
        std::string name;
        std::string filename;
        std::string contentType;
        std::string data;
    };

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

    static inline std::string trimStr(const std::string& s) {
        size_t b = 0, e = s.size();
        while (b < e && (s[b] == ' ' || s[b] == '\t' ||
            s[b] == '\r' || s[b] == '\n')) ++b;
        while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' ||
            s[e - 1] == '\r' || s[e - 1] == '\n')) --e;
        return s.substr(b, e - b);
    }

    static inline std::string toLower(std::string s) {
        for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
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

    static inline bool writeFile(const fs::path& p, const std::string& data) {
        std::ofstream f(p, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        f.write(data.data(), static_cast<std::streamsize>(data.size()));
        return static_cast<bool>(f);
    }

    static inline std::string makeSettings(const CreateParams& p) {
        std::string j;
        j.reserve(220);
        j += "{\n";
        j += "\t\"useGPU\":" + std::string(p.useGPU ? "true" : "false") + ",\n";
        j += "\t\"modelSelected\":\"" + p.modelSelected + "\",\n";
        j += "\t\"imgsz\":" + std::to_string(p.imgsz) + ",\n";
        j += "\t\"lr\":" + std::to_string(p.lr) + ",\n";
        j += "\t\"batchSize\":" + std::to_string(p.batchSize) + ",\n";
        j += "\t\"worker\":" + std::to_string(p.worker) + "\n";
        j += "}\n";
        return j;
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
            if (next == std::string::npos) {
                return false;
            }

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

    static inline bool parseBool(const std::string& s, bool& out) {
        std::string v = toLower(trimStr(s));
        if (v == "true" || v == "1") { out = true;  return true; }
        if (v == "false" || v == "0") { out = false; return true; }
        return false;
    }

    static inline bool parseInt(const std::string& s, long long& out) {
        try {
            size_t idx = 0;
            long long v = std::stoll(trimStr(s), &idx);
            if (idx != trimStr(s).size()) return false;
            out = v;
            return true;
        }
        catch (...) { return false; }
    }

    static inline bool parseDouble(const std::string& s, double& out) {
        try {
            size_t idx = 0;
            double v = std::stod(trimStr(s), &idx);
            if (idx != trimStr(s).size()) return false;
            out = v;
            return true;
        }
        catch (...) { return false; }
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

        std::string login, password, name, type, stateStr;
        std::string useGPUStr, modelSelected, imgszStr, lrStr, batchSizeStr, workerStr;
        std::string cocoData;
        bool hasCoco = false;

        for (auto& p : parts) {
            if (p.name == "login")         login = p.data;
            else if (p.name == "password")      password = p.data;
            else if (p.name == "name")          name = p.data;
            else if (p.name == "type")          type = p.data;
            else if (p.name == "state")         stateStr = p.data;
            else if (p.name == "coco") { cocoData = p.data; hasCoco = true; }
            else if (p.name == "useGPU")        useGPUStr = p.data;
            else if (p.name == "modelSelected") modelSelected = p.data;
            else if (p.name == "imgsz")         imgszStr = p.data;
            else if (p.name == "lr")            lrStr = p.data;
            else if (p.name == "batchSize")     batchSizeStr = p.data;
            else if (p.name == "worker")        workerStr = p.data;
        }

        login = trimStr(login);
        password = trimStr(password);
        name = trimStr(name);
        type = trimStr(type);

        if (login.empty())    return mkError("login required");
        if (password.empty()) return mkError("password required");
        if (name.empty())     return mkError("name required");
        if (type.empty())     return mkError("type required");
        if (stateStr.empty()) return mkError("state required");
        if (!hasCoco)         return mkError("coco file required");

        CreateParams p;
        p.name = name;
        p.type = type;
        if (!parseBool(stateStr, p.state)) {
            return mkError("state invalid");
        }

        if (p.state) {
            if (!parseBool(useGPUStr, p.useGPU)) {
                return mkError("useGPU invalid");
            }
            p.modelSelected = trimStr(modelSelected);
            if (p.modelSelected != "yolov8n" && p.modelSelected != "yolov8s" &&
                p.modelSelected != "yolov8m" && p.modelSelected != "yolov8l" &&
                p.modelSelected != "yolov8x") {
                return mkError("modelSelected invalid");
            }
            if (!parseInt(imgszStr, p.imgsz)) {
                return mkError("imgsz invalid");
            }
            if (p.imgsz < 32 || p.imgsz > 4096 || (p.imgsz % 32) != 0) {
                return mkError("imgsz must be 32..4096 multiple of 32");
            }
            if (!parseDouble(lrStr, p.lr)) {
                return mkError("lr invalid");
            }
            if (!(p.lr > 0.0 && p.lr <= 1.0)) {
                return mkError("lr must be in (0;1]");
            }
            if (!parseInt(batchSizeStr, p.batchSize)) {
                return mkError("batchSize invalid");
            }
            if (p.batchSize < 1 || p.batchSize > 512) {
                return mkError("batchSize must be 1..512");
            }
            if (!parseInt(workerStr, p.worker)) {
                return mkError("worker invalid");
            }
            if (p.worker < 0 || p.worker > 32) {
                return mkError("worker must be 0..32");
            }
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

        if (!isSafeName(p.name)) {
            return mkError("invalid name");
        }

        std::error_code ec;
        fs::create_directories(kProjectsRoot, ec);
        if (ec) {
            return mkError("cannot create projects dir");
        }

        fs::path projDir = kProjectsRoot / p.name;
        if (fs::exists(projDir, ec)) {
            return mkError("project already exists");
        }

        if (!fs::create_directory(projDir, ec) || ec) {
            return mkError("cannot create project dir");
        }
        if (!fs::create_directory(projDir / "images", ec) || ec) {
            return mkError("cannot create images dir");
        }
        if (!fs::create_directory(projDir / "labels", ec) || ec) {
            return mkError("cannot create labels dir");
        }
        if (!writeFile(projDir / "status.txt", p.state ? "true" : "false")) {
            return mkError("cannot write status.txt");
        }
        if (!writeFile(projDir / "type.txt", p.type)) {
            return mkError("cannot write type.txt");
        }
        if (!writeFile(projDir / "coco.txt", cocoData)) {
            return mkError("cannot write coco.txt");
        }
        if (!writeFile(projDir / "modelSettings.json", makeSettings(p))) {
            return mkError("cannot write modelSettings.json");
        }

        if (p.state && g_trainEnqueue) {
            g_trainEnqueue(projDir);
        }

        return mkJson(http::status::ok, "{\"success\":true}");
    }

}