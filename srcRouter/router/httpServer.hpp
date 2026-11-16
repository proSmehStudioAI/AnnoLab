#pragma once
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
namespace ssl = net::ssl;
using tcp = net::ip::tcp;
namespace fs = std::filesystem;

class httpHttpsServer {
public:
    struct ApiResponse {
        http::status status = http::status::ok;
        std::string  contentType = "application/json; charset=utf-8";
        std::string  body;
        std::string  contentDisposition;
    };
    using ApiHandler = std::function<ApiResponse(
        const std::string& clientIp,
        const std::string& pathAfterApi,
        const http::request<http::string_body>& req)>;

    httpHttpsServer(std::string ip,
        unsigned short portHttp,
        unsigned short portHttps,
        std::string serverKey,
        std::string serverCrt,
        fs::path staticRoot = "static")
        : ip_(std::move(ip))
        , portHttp_(portHttp)
        , portHttps_(portHttps)
        , serverKey_(std::move(serverKey))
        , serverCrt_(std::move(serverCrt))
        , root_(std::move(staticRoot))
    {
        std::error_code ec;
        fs::create_directories(root_, ec);
    }

    ~httpHttpsServer() { stop(); }

    httpHttpsServer(const httpHttpsServer&) = delete;
    httpHttpsServer& operator=(const httpHttpsServer&) = delete;

    void addApiHandler(std::string prefix, ApiHandler handler) {
        std::lock_guard<std::mutex> lk(apiRegistry_->mtx);
        apiRegistry_->byPrefix[std::move(prefix)] = std::move(handler);
    }

    void removeApiHandler(const std::string& prefix) {
        std::lock_guard<std::mutex> lk(apiRegistry_->mtx);
        apiRegistry_->byPrefix.erase(prefix);
    }

    void clearApiHandlers() {
        std::lock_guard<std::mutex> lk(apiRegistry_->mtx);
        apiRegistry_->byPrefix.clear();
    }

    void stop() {
        if (!running_->exchange(false)) return;
        cv_.notify_all();
        beast::error_code ec;
        acceptor_.cancel(ec);
        acceptorS_.cancel(ec);
    }

    int run() {
        try {
            setup();
        }
        catch (const std::exception& e) {
            std::cerr << "Setup error: " << e.what() << '\n';
            return -1;
        }

        startAcceptHttp();
        startAcceptHttps();

        tHttp_ = std::thread([this] { ioc_->run(); });
        tHttps_ = std::thread([this] { iocS_->run(); });

        std::cout << "HTTP  server: http://" << ip_ << ':' << portHttp_ << '\n'
            << "HTTPS server: https://" << ip_ << ':' << portHttps_ << '\n'
            << "Serving: " << fs::absolute(root_).string() << '\n';

        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait(lk, [&] { return !running_->load(); });
        }

        if (tHttp_.joinable())  tHttp_.join();
        if (tHttps_.joinable()) tHttps_.join();

        {
            std::unique_lock<std::mutex> lk(sessions_->mtx);
            sessions_->cv.wait_for(lk, std::chrono::seconds(35),
                [&] { return sessions_->count.load() == 0; });
        }
        return 0;
    }
    static std::unordered_map<std::string, std::string>
        parseQuery(const std::string& target) {
        std::unordered_map<std::string, std::string> out;
        auto q = target.find('?');
        if (q == std::string::npos) return out;

        std::string qs = target.substr(q + 1);
        size_t pos = 0;
        while (pos <= qs.size()) {
            auto amp = qs.find('&', pos);
            std::string pair = qs.substr(pos, amp == std::string::npos
                ? std::string::npos
                : amp - pos);
            if (!pair.empty()) {
                auto eq = pair.find('=');
                std::string key = eq == std::string::npos ? pair : pair.substr(0, eq);
                std::string value = eq == std::string::npos ? std::string{}
                : pair.substr(eq + 1);
                out[urlDecode(key)] = urlDecode(value);
            }
            if (amp == std::string::npos) break;
            pos = amp + 1;
        }
        return out;
    }
private:
    std::string ip_;
    unsigned short portHttp_;
    unsigned short portHttps_;
    std::string serverKey_;
    std::string serverCrt_;
    fs::path root_;

    struct SessionCounters {
        std::atomic<int> count{ 0 };
        std::mutex mtx;
        std::condition_variable cv;
    };

    struct ApiRegistry {
        std::mutex mtx;
        std::map<std::string, ApiHandler> byPrefix;
    };

    std::shared_ptr<net::io_context>   ioc_ = std::make_shared<net::io_context>();
    std::shared_ptr<net::io_context>   iocS_ = std::make_shared<net::io_context>();
    std::shared_ptr<ssl::context>      sslCtx_ = std::make_shared<ssl::context>(ssl::context::tls_server);
    std::shared_ptr<std::atomic<bool>> running_ = std::make_shared<std::atomic<bool>>(true);
    std::shared_ptr<SessionCounters>   sessions_ = std::make_shared<SessionCounters>();
    std::shared_ptr<ApiRegistry>       apiRegistry_ = std::make_shared<ApiRegistry>();

    tcp::acceptor acceptor_{ *ioc_ };
    tcp::acceptor acceptorS_{ *iocS_ };

    std::thread tHttp_;
    std::thread tHttps_;

    std::mutex mtx_;
    std::condition_variable cv_;

    void setup() {
        sslCtx_->set_options(ssl::context::default_workarounds |
            ssl::context::no_sslv2 |
            ssl::context::no_sslv3);
        sslCtx_->use_certificate_chain_file(serverCrt_);
        sslCtx_->use_private_key_file(serverKey_, ssl::context::pem);

        const auto addr = net::ip::make_address(ip_);
        const auto proto = addr.is_v4() ? tcp::v4() : tcp::v6();

        acceptor_.open(proto);
        acceptor_.set_option(net::socket_base::reuse_address(true));
        acceptor_.bind({ addr, portHttp_ });
        acceptor_.listen();

        acceptorS_.open(proto);
        acceptorS_.set_option(net::socket_base::reuse_address(true));
        acceptorS_.bind({ addr, portHttps_ });
        acceptorS_.listen();
    }

    void startAcceptHttp() {
        acceptor_.async_accept(
            [this](beast::error_code ec, tcp::socket socket) {
                if (!ec) {
                    sessions_->count.fetch_add(1);
                    auto root = root_;
                    auto running = running_;
                    auto iocKeep = ioc_;
                    auto cnt = sessions_;
                    auto apiReg = apiRegistry_;
                    std::thread([s = std::move(socket), root, running,
                        iocKeep, cnt, apiReg]() mutable {
                            struct G {
                                std::shared_ptr<SessionCounters> c;
                                ~G() { c->count.fetch_sub(1); c->cv.notify_all(); }
                            } g{ cnt };
                            runPlainSession(std::move(s), root, running, iocKeep, apiReg);
                        }).detach();
                }
                if (running_->load()) startAcceptHttp();
            });
    }

    void startAcceptHttps() {
        acceptorS_.async_accept(
            [this](beast::error_code ec, tcp::socket socket) {
                if (!ec) {
                    sessions_->count.fetch_add(1);
                    auto root = root_;
                    auto running = running_;
                    auto iocKeep = iocS_;
                    auto sslKeep = sslCtx_;
                    auto cnt = sessions_;
                    auto apiReg = apiRegistry_;
                    std::thread([s = std::move(socket), root, running,
                        iocKeep, sslKeep, cnt, apiReg]() mutable {
                            struct G {
                                std::shared_ptr<SessionCounters> c;
                                ~G() { c->count.fetch_sub(1); c->cv.notify_all(); }
                            } g{ cnt };
                            beast::ssl_stream<beast::tcp_stream> stream(std::move(s), *sslKeep);
                            runTlsSession(std::move(stream), root, running, iocKeep, sslKeep, apiReg);
                        }).detach();
                }
                if (running_->load()) startAcceptHttps();
            });
    }

    static void runPlainSession(tcp::socket socket,
        fs::path root,
        std::shared_ptr<std::atomic<bool>> running,
        std::shared_ptr<net::io_context>,
        std::shared_ptr<ApiRegistry> apiReg) {
        std::string clientIp;
        {
            beast::error_code epEc;
            auto ep = socket.remote_endpoint(epEc);
            if (!epEc) clientIp = ep.address().to_string();
        }

        beast::tcp_stream stream(std::move(socket));
        beast::flat_buffer buffer;
        beast::error_code ec;

        while (running->load()) {
            stream.expires_after(std::chrono::seconds(30));
            http::request<http::string_body> req;
            http::read(stream, buffer, req, ec);
            if (ec == http::error::end_of_stream || ec == net::error::eof) break;
            if (ec) return;

            http::response<http::string_body> res;
            handleRequest(req, res, root, clientIp, apiReg);
            res.keep_alive(req.keep_alive());

            http::write(stream, res, ec);
            if (ec) return;
            if (!res.keep_alive()) break;
        }

        stream.socket().shutdown(tcp::socket::shutdown_send, ec);
    }

    static void runTlsSession(beast::ssl_stream<beast::tcp_stream> stream,
        fs::path root,
        std::shared_ptr<std::atomic<bool>> running,
        std::shared_ptr<net::io_context>,
        std::shared_ptr<ssl::context>,
        std::shared_ptr<ApiRegistry> apiReg) {
        beast::error_code ec;
        beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
        stream.handshake(ssl::stream_base::server, ec);
        if (ec) return;

        std::string clientIp;
        {
            beast::error_code epEc;
            auto ep = beast::get_lowest_layer(stream).socket().remote_endpoint(epEc);
            if (!epEc) clientIp = ep.address().to_string();
        }

        beast::flat_buffer buffer;
        while (running->load()) {
            beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(30));
            http::request<http::string_body> req;
            http::read(stream, buffer, req, ec);
            if (ec == http::error::end_of_stream || ec == net::error::eof) break;
            if (ec) return;

            http::response<http::string_body> res;
            handleRequest(req, res, root, clientIp, apiReg);
            res.keep_alive(req.keep_alive());

            http::write(stream, res, ec);
            if (ec) return;
            if (!res.keep_alive()) break;
        }

        beast::get_lowest_layer(stream).expires_after(std::chrono::seconds(2));
        stream.shutdown(ec);
    }

    static std::string mimeType(const fs::path& p) {
        static const std::unordered_map<std::string, std::string> m = {
            {".html","text/html; charset=utf-8"},
            {".htm", "text/html; charset=utf-8"},
            {".css", "text/css; charset=utf-8"},
            {".js",  "application/javascript; charset=utf-8"},
            {".mjs", "application/javascript; charset=utf-8"},
            {".json","application/json; charset=utf-8"},
            {".xml", "application/xml; charset=utf-8"},
            {".txt", "text/plain; charset=utf-8"},
            {".png", "image/png"},
            {".jpg", "image/jpeg"},
            {".jpeg","image/jpeg"},
            {".gif", "image/gif"},
            {".svg", "image/svg+xml"},
            {".webp","image/webp"},
            {".ico", "image/x-icon"},
            {".woff","font/woff"},
            {".woff2","font/woff2"},
            {".ttf", "font/ttf"},
            {".pdf", "application/pdf"},
            {".wasm","application/wasm"},
            {".map", "application/json"},
        };
        std::string ext = p.extension().string();
        for (auto& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        auto it = m.find(ext);
        return it != m.end() ? it->second : "application/octet-stream";
    }

    static std::string urlDecode(const std::string& in) {
        std::string out;
        out.reserve(in.size());
        for (size_t i = 0; i < in.size(); ++i) {
            if (in[i] == '%' && i + 2 < in.size()) {
                char hex[3] = { in[i + 1], in[i + 2], 0 };
                char* end = nullptr;
                long v = std::strtol(hex, &end, 16);
                if (end == hex + 2) {
                    out.push_back(static_cast<char>(v));
                    i += 2;
                    continue;
                }
            }
            out.push_back(in[i] == '+' ? ' ' : in[i]);
        }
        return out;
    }

    static std::optional<fs::path> resolvePath(const std::string& target, const fs::path& root) {
        std::string decoded = urlDecode(target);
        auto q = decoded.find('?'); if (q != std::string::npos) decoded.resize(q);
        auto h = decoded.find('#'); if (h != std::string::npos) decoded.resize(h);

        if (decoded.empty() || decoded[0] != '/') return std::nullopt;
        decoded.erase(0, 1);
        if (decoded.find('\0') != std::string::npos) return std::nullopt;

        std::error_code ec;
        fs::path canon = fs::weakly_canonical(root / decoded, ec);
        if (ec) return std::nullopt;
        fs::path canonRoot = fs::weakly_canonical(root, ec);
        if (ec) return std::nullopt;

        auto rel = canon.lexically_relative(canonRoot);
        if (rel.empty()) return std::nullopt;
        if (*rel.begin() == "..") return std::nullopt;

        return canon;
    }

    static void sendError(http::response<http::string_body>& res,
        http::status status,
        std::string msg,
        const std::string& allow = "") {
        res.result(status);
        res.set(http::field::content_type, "text/plain; charset=utf-8");
        if (!allow.empty()) res.set(http::field::allow, allow);
        res.body() = std::move(msg);
        res.prepare_payload();
    }

    static bool tryHandleApi(const http::request<http::string_body>& req,
        http::response<http::string_body>& res,
        const std::string& clientIp,
        const std::shared_ptr<ApiRegistry>& apiReg) {
        std::string target(req.target());
        std::string pathOnly = target;
        auto q = pathOnly.find('?');
        if (q != std::string::npos) pathOnly.resize(q);

        std::string decoded = urlDecode(pathOnly);

        if (decoded != "/api" && decoded.compare(0, 5, "/api/") != 0) return false;

        std::string pathAfterApi = decoded.size() > 5 ? decoded.substr(5) : std::string{};

        ApiHandler handler;
        {
            std::lock_guard<std::mutex> lk(apiReg->mtx);
            size_t bestLen = 0;
            bool   found = false;
            for (auto& [prefix, h] : apiReg->byPrefix) {
                if (pathAfterApi.size() >= prefix.size() &&
                    pathAfterApi.compare(0, prefix.size(), prefix) == 0 &&
                    (!found || prefix.size() >= bestLen)) {
                    bestLen = prefix.size();
                    handler = h;
                    found = true;
                }
            }
        }

        if (!handler) {
            sendError(res, http::status::not_found, "No API handler for this path");
            return true;
        }

        ApiResponse apiRes;
        try {
            apiRes = handler(clientIp, pathAfterApi, req);
        }
        catch (const std::exception& e) {
            sendError(res, http::status::internal_server_error,
                std::string("API error: ") + e.what());
            return true;
        }
        catch (...) {
            sendError(res, http::status::internal_server_error, "API unknown error");
            return true;
        }
        res.result(apiRes.status);
        res.set(http::field::content_type, apiRes.contentType);
        res.set(http::field::cache_control, "no-cache");

        if (!apiRes.contentDisposition.empty()) {
            res.set(http::field::content_disposition, apiRes.contentDisposition);
        }

        if (req.method() == http::verb::head) {
            res.content_length(apiRes.body.size());
        }
        else {
            res.body() = std::move(apiRes.body);
            res.prepare_payload();
        }
        return true;
    }

    static void handleRequest(const http::request<http::string_body>& req,
        http::response<http::string_body>& res,
        const fs::path& root,
        const std::string& clientIp,
        const std::shared_ptr<ApiRegistry>& apiReg) {
        res.version(req.version());
        if (tryHandleApi(req, res, clientIp, apiReg)) return;
        if (req.method() != http::verb::get && req.method() != http::verb::head) {
            sendError(res, http::status::method_not_allowed,
                "Method Not Allowed", "GET, HEAD");
            return;
        }

        auto pathOpt = resolvePath(std::string(req.target()), root);
        if (!pathOpt) {
            sendError(res, http::status::forbidden, "Forbidden");
            return;
        }

        fs::path p = *pathOpt;
        std::error_code ec;
        if (fs::is_directory(p, ec)) p /= "index.html";
        if (!fs::is_regular_file(p, ec)) {
            sendError(res, http::status::not_found, "Not Found");
            return;
        }

        std::ifstream f(p, std::ios::binary);
        if (!f) {
            sendError(res, http::status::internal_server_error, "Internal Server Error");
            return;
        }
        std::string content((std::istreambuf_iterator<char>(f)),
            std::istreambuf_iterator<char>());

        res.result(http::status::ok);
        res.set(http::field::content_type, mimeType(p));
        res.set(http::field::cache_control, "no-cache");

        if (req.method() == http::verb::head) {
            res.content_length(content.size());
        }
        else {
            res.body() = std::move(content);
            res.prepare_payload();
        }
    }
};