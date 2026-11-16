#pragma once
#include "httpServer.hpp"
#include "globals.hpp"

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>

namespace login_event {

    constexpr int  kMaxRequestsPerMinute = 10;
    constexpr auto kWindow = std::chrono::seconds(60);
    constexpr auto kBlockDuration = std::chrono::seconds(60);

    struct RateState {
        std::chrono::steady_clock::time_point windowStart{};
        int                                   count = 0;
        std::chrono::steady_clock::time_point blockedUntil{};
    };

    inline std::mutex                                 g_rateMtx;
    inline std::unordered_map<std::string, RateState> g_rate;
    inline std::mutex                                 g_accountMtx;

    inline httpHttpsServer::ApiResponse
        handle(const std::string& ip,
            const std::string& /*rest*/,
            const http::request<http::string_body>& req)
    {
        using clock = std::chrono::steady_clock;
        const auto now = clock::now();

        {
            std::lock_guard<std::mutex> lk(g_rateMtx);
            RateState& st = g_rate[ip];

            if (st.blockedUntil > now) {
                auto remain = std::chrono::duration_cast<std::chrono::seconds>(
                    st.blockedUntil - now).count();
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
            const std::string& login = itL->second;
            const std::string& password = itP->second;

            std::lock_guard<std::mutex> lk(g_accountMtx);
            for (const auto& [l, p] : globals::accaunt) {
                if (l == login && p == password) { ok = true; break; }
            }
        }

        if (ok) {
            httpHttpsServer::ApiResponse r;
            r.contentType = "text/plain; charset=utf-8";
            r.body = "true";
            return r;
        }

        {
            std::lock_guard<std::mutex> lk(g_rateMtx);
            RateState& st = g_rate[ip];

            if (now - st.windowStart >= kWindow) {
                st.windowStart = now;
                st.count = 0;
            }

            ++st.count;
            if (st.count > kMaxRequestsPerMinute) {
                st.blockedUntil = now + kBlockDuration;
                auto remain = std::chrono::duration_cast<std::chrono::seconds>(
                    st.blockedUntil - now).count();

                httpHttpsServer::ApiResponse r;
                r.status = http::status::too_many_requests;
                r.contentType = "application/json; charset=utf-8";
                r.body = "{\"pause\":" + std::to_string(remain) + "}";
                return r;
            }
        }

        httpHttpsServer::ApiResponse r;
        r.contentType = "text/plain; charset=utf-8";
        r.body = "false";
        return r;
    }

}