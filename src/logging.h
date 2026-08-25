#pragma once

// Logging frontend for the checkpoint daemon.

#include <fmt/core.h>
#include <fmt/format.h>

#include <cstdio>

namespace checkpoint::logging
{

inline void Shutdown() noexcept
{
    std::fflush(stdout);
    std::fflush(stderr);
}

}    // namespace checkpoint::logging

// NOLINTBEGIN(cppcoreguidelines-macro-usage)

#define INFO(...) fmt::println(stderr, "[native_ckpt:svr][INFO] " __VA_ARGS__)
#define WARN(...) fmt::println(stderr, "[native_ckpt:svr][WARN] " __VA_ARGS__)
#define ERR(...) fmt::println(stderr, "[native_ckpt:svr][ERROR] " __VA_ARGS__)
#define VERBOSE(...) fmt::println(stderr, "[native_ckpt:svr][DEBUG] " __VA_ARGS__)
#define CRIT(...) fmt::println(stderr, "[native_ckpt:svr][CRITICAL] " __VA_ARGS__)

// NOLINTEND(cppcoreguidelines-macro-usage)
