#pragma once

#include <fmt/core.h>
#include <fmt/format.h>

#include <cstdint>
#include <iterator>
#include <string_view>
#include <utility>

namespace checkpoint::logging
{

enum class Level : uint8_t
{
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Critical,
};

void Init();

void Flush();

void Shutdown();

void LogImpl(Level level, std::string_view message);

template <typename... Args> void Log(Level level, fmt::format_string<Args...> format, Args&&... args)
{
    fmt::memory_buffer buffer;
    fmt::format_to(std::back_inserter(buffer), format, std::forward<Args>(args)...);
    LogImpl(level, std::string_view{buffer.data(), buffer.size()});
}

template <typename... Args> void Trace(fmt::format_string<Args...> format, Args&&... args)
{ Log(Level::Trace, format, std::forward<Args>(args)...); }
template <typename... Args> void Debug(fmt::format_string<Args...> format, Args&&... args)
{ Log(Level::Debug, format, std::forward<Args>(args)...); }
template <typename... Args> void Info(fmt::format_string<Args...> format, Args&&... args)
{ Log(Level::Info, format, std::forward<Args>(args)...); }
template <typename... Args> void Warn(fmt::format_string<Args...> format, Args&&... args)
{ Log(Level::Warn, format, std::forward<Args>(args)...); }
template <typename... Args> void Error(fmt::format_string<Args...> format, Args&&... args)
{ Log(Level::Error, format, std::forward<Args>(args)...); }
template <typename... Args> void Critical(fmt::format_string<Args...> format, Args&&... args)
{ Log(Level::Critical, format, std::forward<Args>(args)...); }

}    // namespace checkpoint::logging

// NOLINTBEGIN(cppcoreguidelines-macro-usage)

#define INFO(...) ::checkpoint::logging::Info(__VA_ARGS__)
#define WARN(...) ::checkpoint::logging::Warn(__VA_ARGS__)
#define ERR(...) ::checkpoint::logging::Error(__VA_ARGS__)
#define VERBOSE(...) ::checkpoint::logging::Debug(__VA_ARGS__)
#define CRIT(...) ::checkpoint::logging::Critical(__VA_ARGS__)

// NOLINTEND(cppcoreguidelines-macro-usage)
