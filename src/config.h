#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

#include <unistd.h>

namespace checkpoint::config
{
[[nodiscard]] inline std::string Hostname()
{
    std::array<char, 256> value{};
    if (::gethostname(value.data(), value.size() - 1) != 0) { return "unknown-host"; }
    return value.data();
}

[[nodiscard]] inline std::string Substitute(std::string_view value)
{
    std::string result{value};
    auto        replace = [&result](std::string_view token, std::string const& replacement) {
        for (size_t pos = result.find(token); pos != std::string::npos; pos = result.find(token, pos + replacement.size()))
        {
            result.replace(pos, token.size(), replacement);
        }
    };
    if (result.contains("%h")) { replace("%h", Hostname()); }
    if (result.contains("%p")) { replace("%p", std::to_string(::getpid())); }
    return result;
}
}    // namespace checkpoint::config

namespace checkpoint::config::types
{
struct Text
{
    using ValueType = std::string;

    static ValueType Parse(std::string_view value) { return std::string{value}; }
};

struct OptIn
{
    using ValueType = bool;

    static ValueType Parse(std::string_view value)
    { return value == "1" || value == "true" || value == "TRUE" || value == "on" || value == "ON" || value == "yes"; }
};

struct OptOut
{
    using ValueType = bool;

    static ValueType Parse(std::string_view value)
    { return !(value == "0" || value == "false" || value == "off" || value == "OFF" || value == "no"); }
};

struct EnabledByOne
{
    using ValueType = bool;

    static ValueType Parse(std::string_view value) { return value == "1"; }
};

struct RFile
{
    using ValueType = std::filesystem::path;

    static ValueType Parse(std::string_view value)
    {
        ValueType const path{Substitute(value)};
        if (!std::filesystem::exists(path)) { throw std::runtime_error("File does not exist: " + path.string()); }
        return path;
    }
};

struct WFile
{
    using ValueType = std::filesystem::path;

    static ValueType Parse(std::string_view value) { return ValueType{Substitute(value)}; }
};

struct WDir
{
    using ValueType = std::filesystem::path;

    static ValueType Parse(std::string_view value)
    {
        ValueType const path{Substitute(value)};
        if (!path.empty()) { static_cast<void>(std::filesystem::create_directories(path)); }
        return path;
    }
};

struct OtelExportMode
{
    enum class ValueType : uint8_t
    {
        File,
        Daemon,
        Both
    };

    static ValueType Parse(std::string_view value)
    {
        if (value == "file") { return ValueType::File; }
        if (value == "both") { return ValueType::Both; }
        return ValueType::Daemon;
    }
};
}    // namespace checkpoint::config::types

namespace checkpoint::config
{
struct LogLevel
{
    static constexpr std::string_view Name    = "log.level";
    static constexpr std::string_view Default = "info";
    using Type                                = types::Text;
};

struct ServerLog
{
    static constexpr std::string_view Name    = "server.log";
    static constexpr std::string_view Default = "/tmp/native_ckpt_server.%h_%p.log";
    using Type                                = types::WFile;
};

struct SvcDisableAutoLaunch
{
    static constexpr std::string_view Env     = "CHECKPOINT_SVC_DISABLE_AUTO_LAUNCH";
    static constexpr std::string_view Default = "0";
    using Type                                = types::EnabledByOne;
};

struct OtelEnable
{
    static constexpr std::string_view Name    = "otel.enable";
    static constexpr std::string_view Default = "0";
    using Type                                = types::OptIn;
};

struct OtelExport
{
    static constexpr std::string_view Name    = "otel.export";
    static constexpr std::string_view Default = "daemon";
    using Type                                = types::OtelExportMode;
};

struct OtelFile
{
    static constexpr std::string_view Name = "otel.file";
    static constexpr std::string_view Default;
    using Type = types::WFile;
};

struct OtelDir
{
    static constexpr std::string_view Name    = "otel.dir";
    static constexpr std::string_view Default = "/tmp";
    using Type                                = types::WDir;
};

struct OtelService
{
    static constexpr std::string_view Name = "otel.service";
    static constexpr std::string_view Default;
    using Type = types::Text;
};

struct OtelCollector
{
    static constexpr std::string_view Name = "otel.collector";
    static constexpr std::string_view Default;
    using Type = types::Text;
};

struct OtelRunId
{
    static constexpr std::string_view Env = "MEGATRON_OTEL_RUN_ID";
    static constexpr std::string_view Default;
    using Type = types::Text;
};

struct CiPhase
{
    static constexpr std::string_view Name = "ci.phase";
    static constexpr std::string_view Default;
    using Type = types::Text;
};

struct OtelMetrics
{
    static constexpr std::string_view Name    = "otel.metrics";
    static constexpr std::string_view Default = "0";
    using Type                                = types::OptIn;
};

struct OtelLogs
{
    static constexpr std::string_view Name    = "otel.logs";
    static constexpr std::string_view Default = "1";
    using Type                                = types::OptOut;
};

namespace detail
{
static constexpr std::string_view EnvPrefix = "NATIVE_CKPT_";

static constexpr char EnvChar(char value)
{
    if (value == '.') { return '_'; }
    return value >= 'a' && value <= 'z' ? static_cast<char>(value - 'a' + 'A') : value;
}

template <typename TConfig> consteval auto EnvName()
{
    if constexpr (requires { TConfig::Env; })
    {
        std::array<char, TConfig::Env.size() + 1> name{};
        size_t                                    index = 0;
        for (char const value : TConfig::Env) { name[index++] = value; }
        return name;
    }
    else
    {
        std::array<char, EnvPrefix.size() + TConfig::Name.size() + 1> name{};
        size_t                                                        index = 0;
        for (char const value : EnvPrefix) { name[index++] = value; }
        for (char const value : TConfig::Name) { name[index++] = EnvChar(value); }
        return name;
    }
}
}    // namespace detail

template <typename TConfig> [[nodiscard]] auto GetConfigValue()
{
    constexpr auto         envName  = detail::EnvName<TConfig>();
    char const*            rawValue = std::getenv(envName.data());    // NOLINT(concurrency-mt-unsafe)
    std::string_view const value    = rawValue != nullptr && *rawValue != '\0' ? std::string_view{rawValue} : TConfig::Default;
    return TConfig::Type::Parse(value);
}

using OtelExportMode = types::OtelExportMode::ValueType;

struct OtelConfig
{
    bool                  enabled   = GetConfigValue<OtelEnable>();
    OtelExportMode        mode      = GetConfigValue<OtelExport>();
    std::filesystem::path file      = GetConfigValue<OtelFile>();
    std::filesystem::path dir       = GetConfigValue<OtelDir>();
    std::string           service   = GetConfigValue<OtelService>();
    std::string           collector = GetConfigValue<OtelCollector>();
    std::string           runId     = GetConfigValue<OtelRunId>();
    std::string           ciPhase   = GetConfigValue<CiPhase>();
    bool                  metrics   = GetConfigValue<OtelMetrics>();
    bool                  logs      = GetConfigValue<OtelLogs>();

    [[nodiscard]] std::filesystem::path ExportFile() const
    { return file.empty() ? dir / Substitute("native_ckpt_otel_svr_%h_pid%p.jsonl") : file; }

    [[nodiscard]] std::filesystem::path CollectorFile() const
    { return file.empty() ? std::filesystem::path{"/tmp/native_ckpt_otel.jsonl"} : file; }
};
}    // namespace checkpoint::config
