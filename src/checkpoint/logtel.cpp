#include "CommonMacros.h"
#include "SetThreadName.h"

#include "config.h"
#include "logging.h"
#include "net_utils.h"
#include "process_memory.h"
#include "telemetry.h"

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/container/small_vector.hpp>

#include <fmt/format.h>

#include <spdlog/async.h>
#include <spdlog/sinks/sink.h>
#include <spdlog/spdlog.h>

#include <opentelemetry/common/key_value_iterable_view.h>
#include <opentelemetry/context/context.h>
#include <opentelemetry/exporters/otlp/otlp_file_client.h>
#include <opentelemetry/exporters/otlp/otlp_file_exporter_factory.h>
#include <opentelemetry/exporters/otlp/otlp_file_log_record_exporter_factory.h>
#include <opentelemetry/exporters/otlp/otlp_file_metric_exporter_factory.h>
#include <opentelemetry/logs/provider.h>
#include <opentelemetry/metrics/provider.h>
#include <opentelemetry/proto/collector/logs/v1/logs_service.pb.h>
#include <opentelemetry/proto/collector/metrics/v1/metrics_service.pb.h>
#include <opentelemetry/proto/collector/trace/v1/trace_service.pb.h>
#include <opentelemetry/sdk/logs/batch_log_record_processor_factory.h>
#include <opentelemetry/sdk/logs/logger_provider_factory.h>
#include <opentelemetry/sdk/metrics/export/periodic_exporting_metric_reader_factory.h>
#include <opentelemetry/sdk/metrics/meter_provider_factory.h>
#include <opentelemetry/sdk/trace/batch_span_processor_factory.h>
#include <opentelemetry/sdk/trace/tracer_provider_factory.h>
#include <opentelemetry/trace/provider.h>

#include <expected>
#include <mutex>
#include <sstream>
#include <utility>
#include <variant>

#include <unistd.h>
#include <zlib.h>

namespace trace_api     = opentelemetry::trace;
namespace trace_sdk     = opentelemetry::sdk::trace;
namespace metrics_api   = opentelemetry::metrics;
namespace metrics_sdk   = opentelemetry::sdk::metrics;
namespace logs_api      = opentelemetry::logs;
namespace logs_sdk      = opentelemetry::sdk::logs;
namespace otlp          = opentelemetry::exporter::otlp;
namespace otel_res      = opentelemetry::sdk::resource;
namespace otel_common   = opentelemetry::common;
namespace proto_logs    = opentelemetry::proto::collector::logs::v1;
namespace proto_metrics = opentelemetry::proto::collector::metrics::v1;
namespace proto_trace   = opentelemetry::proto::collector::trace::v1;

static constexpr uint16_t         HttpOk                 = 200;
static constexpr uint16_t         HttpBadRequest         = 400;
static constexpr uint16_t         HttpNotFound           = 404;
static constexpr uint16_t         HttpUnsupportedMedia   = 415;
static constexpr uint16_t         HttpServiceUnavailable = 503;
static constexpr std::string_view OtlpProtobuf           = "application/x-protobuf";

static checkpoint::telemetry::OtlpResponse OtlpError(uint16_t status, std::string_view const& message)
{
    // The OTLP failure body is google.rpc.Status with only its message field set.
    std::string body;
    body.reserve(message.size() + 8);
    body.push_back('\x12');
    size_t length = message.size();
    while (true)
    {
        auto byte = static_cast<uint8_t>(length & 0x7fU);    // NOLINT
        length >>= 7U;                                       // NOLINT
        if (length != 0) { byte |= 0x80U; }
        body.push_back(static_cast<char>(byte));
        if (length == 0) { break; }
    }
    body.append(message);
    return {.status = status, .body = std::move(body)};
}

static std::expected<std::string, std::string> Gunzip(std::string_view const& payload)
{
    z_stream stream{};
    stream.next_in  = reinterpret_cast<Bytef*>(const_cast<char*>(payload.data()));    // NOLINT
    stream.avail_in = static_cast<uInt>(payload.size());
    int result      = ::inflateInit2(&stream, MAX_WBITS + 16);
    if (result != Z_OK) { return std::unexpected{"failed to initialize gzip decoder"}; }

    auto                                   finish = [&stream] { static_cast<void>(::inflateEnd(&stream)); };
    std::string                            decoded;
    std::array<unsigned char, 64UL * 1024> chunk{};
    while (true)
    {
        stream.next_out  = chunk.data();
        stream.avail_out = static_cast<uInt>(chunk.size());
        result           = ::inflate(&stream, Z_NO_FLUSH);
        if (result != Z_OK && result != Z_STREAM_END)
        {
            std::string error = stream.msg == nullptr ? "invalid gzip OTLP request" : stream.msg;
            finish();
            return std::unexpected{std::move(error)};
        }
        size_t const count = chunk.size() - stream.avail_out;
        if (decoded.size() + count > checkpoint::telemetry::MaxOtlpRequestBytes)
        {
            finish();
            return std::unexpected{"decompressed OTLP request exceeds the 64 MiB limit"};
        }
        decoded.append(reinterpret_cast<char const*>(chunk.data()), count);    // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        if (result == Z_STREAM_END) { break; }
    }
    finish();
    return decoded;
}

static otlp::OtlpFileClient MakeJsonConverter(std::ostream& output)
{
    otlp::OtlpFileClientOptions options;
    options.backend_options = std::ref(output);
    return otlp::OtlpFileClient{std::move(options), {}};
}

namespace
{
class OtlpJsonConverter
{
    public:
    OtlpJsonConverter() : _client(MakeJsonConverter(_output)) {}
    ~OtlpJsonConverter() = default;
    CLASS_DELETE_COPY_AND_MOVE(OtlpJsonConverter);

    std::string Convert(google::protobuf::Message const& message)
    {
        _output.str(std::string{});
        _output.clear();
        if (_client.Export(message, 0) != opentelemetry::sdk::common::ExportResult::kSuccess)
        {
            throw std::runtime_error("Failed to serialize OTLP request");
        }
        std::string json = _output.str();
        while (!json.empty() && (json.back() == '\n' || json.back() == '\r')) { json.pop_back(); }
        return json;
    }

    private:
    std::ostringstream   _output;
    otlp::OtlpFileClient _client;
};

class FileCollector
{
    public:
    explicit FileCollector(std::filesystem::path const& filePath) : _file(filePath, std::ios::out | std::ios::app)
    {
        if (!_file.is_open()) { throw std::runtime_error("Failed to open file: " + filePath.string()); }
        _file.exceptions(std::ios::badbit | std::ios::failbit);
    }

    void Collect(std::string_view const& payload)
    {
        _file.write(payload.data(), static_cast<std::streamsize>(payload.size()));
        _file.put('\n');
        _file.flush();
    }

    checkpoint::telemetry::OtlpResponse
    ReceiveOtlp(std::string_view const& path, std::string_view const& contentEncoding, std::string_view const& payload)
    {
        std::string      decompressed;
        std::string_view protobuf = payload;
        if (boost::beast::iequals(contentEncoding, "gzip"))
        {
            auto decoded = Gunzip(payload);
            if (!decoded) { return OtlpError(HttpBadRequest, decoded.error()); }
            decompressed = std::move(*decoded);
            protobuf     = decompressed;
        }
        else if (!contentEncoding.empty() && !boost::beast::iequals(contentEncoding, "identity"))
        {
            return OtlpError(HttpUnsupportedMedia, fmt::format("unsupported Content-Encoding: {}", contentEncoding));
        }

        if (path == "/v1/traces") { return Receive_<proto_trace::ExportTraceServiceRequest>(protobuf); }
        if (path == "/v1/metrics") { return Receive_<proto_metrics::ExportMetricsServiceRequest>(protobuf); }
        if (path == "/v1/logs") { return Receive_<proto_logs::ExportLogsServiceRequest>(protobuf); }
        return OtlpError(HttpNotFound, fmt::format("unsupported OTLP path: {}", path));
    }

    private:
    template <typename Request> checkpoint::telemetry::OtlpResponse Receive_(std::string_view const& payload)
    {
        Request request;
        if (!request.ParseFromArray(payload.data(), static_cast<int>(payload.size())))
        {
            return OtlpError(HttpBadRequest, "invalid binary protobuf OTLP request");
        }
        if (request.ByteSizeLong() != 0) { Collect(_converter.Convert(request)); }
        return {.status = HttpOk, .body = {}};
    }

    std::ofstream     _file;
    OtlpJsonConverter _converter;
};

class RelayCollector
{
    static constexpr size_t MaxRelayAttempts = 3;

    public:
    explicit RelayCollector(std::string_view const& remote) : _remote(remote)
    { std::tie(_remoteAddr, _results) = ResolveEndpoint(_ioc, _remote); }

    void Collect(std::string_view const& payload)
    {
        for (size_t attempt = 0; attempt < MaxRelayAttempts; ++attempt)
        {
            try
            {
                auto const response = Request_("/api/telemetry/remote", "application/json", {}, payload);
                if (response.result() == boost::beast::http::status::ok) { return; }
                if (attempt + 1 == MaxRelayAttempts)
                {
                    throw std::runtime_error(
                        fmt::format("Failed to relay telemetry to remote={} status={}", _remote, response.result_int()));
                }
            } catch (boost::system::system_error const&)
            {
                if (attempt + 1 == MaxRelayAttempts) { throw; }
            }
        }
    }

    checkpoint::telemetry::OtlpResponse
    ReceiveOtlp(std::string_view const& path, std::string_view const& contentEncoding, std::string_view const& payload)
    try
    {
        auto response = Request_(path, OtlpProtobuf, contentEncoding, payload);
        return {.status = static_cast<uint16_t>(response.result_int()), .body = std::move(response.body())};
    } catch (boost::system::system_error const& error)
    {
        return OtlpError(HttpServiceUnavailable, fmt::format("OTLP relay unavailable: {}", error.what()));
    }

    private:
    boost::beast::http::response<boost::beast::http::string_body> Request_(std::string_view const& path,
                                                                           std::string_view const& contentType,
                                                                           std::string_view const& contentEncoding,
                                                                           std::string_view const& payload)
    {
        static constexpr int                                         Http11 = 11;
        boost::beast::http::request<boost::beast::http::string_body> request{boost::beast::http::verb::post, path, Http11};
        request.set(boost::beast::http::field::host, fmt::format("{}:{}", _remoteAddr.host(), _remoteAddr.port()));
        request.set(boost::beast::http::field::content_type, contentType);
        if (!contentEncoding.empty()) { request.set(boost::beast::http::field::content_encoding, contentEncoding); }
        request.keep_alive(false);
        request.body().assign(payload.data(), payload.size());
        request.prepare_payload();

        boost::beast::tcp_stream stream(_ioc);
        static constexpr auto    RelayTimeout = std::chrono::milliseconds{500};
        stream.expires_after(RelayTimeout);
        stream.connect(_results);
        boost::beast::http::write(stream, request);

        boost::beast::flat_buffer                                     responseBuffer;
        boost::beast::http::response<boost::beast::http::string_body> response;
        boost::beast::http::read(stream, responseBuffer, response);
        return response;
    }

    std::string                                  _remote;
    boost::urls::url                             _remoteAddr;
    boost::asio::io_context                      _ioc{};
    boost::asio::ip::tcp::resolver::results_type _results;
};

struct LogTelState
{
    struct StringHash
    {
        using is_transparent = void;
        [[nodiscard]] size_t operator()(std::string_view const& value) const noexcept { return std::hash<std::string_view>{}(value); }
    };

    using HistogramMap = std::unordered_map<std::string, std::unique_ptr<metrics_api::Histogram<double>>, StringHash, std::equal_to<>>;

    struct MetricState
    {
        std::mutex                                  mutex;
        std::shared_ptr<metrics_sdk::MeterProvider> provider;
        std::shared_ptr<metrics_api::Meter>         meter;
        HistogramMap                                histograms;
    };

    struct LogState
    {
        std::mutex                                mutex;
        std::shared_ptr<logs_sdk::LoggerProvider> provider;
        std::shared_ptr<logs_api::Logger>         logger;
    };

    struct SaveStart
    {
        checkpoint::telemetry::Id id;
        std::chrono::nanoseconds  startedAt;
    };

    struct SaveStartState
    {
        boost::container::small_vector<SaveStart, 16> entries;
    };

    enum class ActiveSpanKind : uint8_t
    {
        SvrFinalize,
        SvrWriteMetadata,
        SvrDurable,
    };

    struct ActiveSpanId
    {
        ActiveSpanKind            kind{};
        checkpoint::telemetry::Id id{};

        bool operator==(ActiveSpanId const&) const = default;
    };

    struct ActiveSpan
    {
        ActiveSpanId                     id;
        std::shared_ptr<trace_api::Span> span;
    };

    struct ActiveSpanState
    {
        boost::container::small_vector<ActiveSpan, 16> entries;
    };

    std::mutex                      mutex;
    checkpoint::telemetry::Id       lastId{};
    std::once_flag                  poolOnce;
    std::shared_ptr<spdlog::logger> logger;

    std::atomic<bool>                          enabled{false};
    std::shared_ptr<trace_sdk::TracerProvider> traceProvider;
    std::shared_ptr<trace_api::Tracer>         tracer;
    MetricState                                metrics;
    LogState                                   logs;
    SaveStartState                             saveStarts;
    ActiveSpanState                            activeSpans;
    checkpoint::telemetry::MemorySampler       sampler;

    std::mutex                                                  collectorMutex;
    std::variant<std::monostate, FileCollector, RelayCollector> collector;
    struct PendingOtlpRequest
    {
        std::string path;
        std::string contentEncoding;
        std::string payload;
    };
    boost::container::small_vector<PendingOtlpRequest, 8> pendingOtlp;
    size_t                                                pendingOtlpBytes{0};

    std::once_flag shutdownOnce;

    void Collect(std::string_view const& payload)
    {
        if (payload.empty()) { return; }
        std::scoped_lock lock(collectorMutex);
        if (auto* file = std::get_if<FileCollector>(&collector)) { file->Collect(payload); }
        else if (auto* relay = std::get_if<RelayCollector>(&collector)) { relay->Collect(payload); }
    }

    template <typename TCollector, typename... Args> void InitCollector(Args&&... args)
    {
        if (!std::holds_alternative<TCollector>(collector)) { collector.emplace<TCollector>(std::forward<Args>(args)...); }
        auto& active = std::get<TCollector>(collector);
        for (auto const& request : pendingOtlp)
        {
            auto const response = active.ReceiveOtlp(request.path, request.contentEncoding, request.payload);
            if (response.status != HttpOk)
            {
                throw std::runtime_error(
                    fmt::format("Failed to drain buffered OTLP request path={} status={}", request.path, response.status));
            }
        }
        pendingOtlp.clear();
        pendingOtlpBytes = 0;
    }

    checkpoint::telemetry::OtlpResponse ReceiveOtlp(std::string_view const& path,
                                                    std::string_view const& contentType,
                                                    std::string_view const& contentEncoding,
                                                    std::string_view const& payload)
    {
        std::scoped_lock lock(collectorMutex);
        if (!boost::beast::iequals(contentType, OtlpProtobuf))
        {
            return OtlpError(HttpUnsupportedMedia, fmt::format("unsupported Content-Type: {}", contentType));
        }
        if (auto* relay = std::get_if<RelayCollector>(&collector)) { return relay->ReceiveOtlp(path, contentEncoding, payload); }
        if (auto* file = std::get_if<FileCollector>(&collector)) { return file->ReceiveOtlp(path, contentEncoding, payload); }
        size_t const requestBytes = path.size() + contentEncoding.size() + payload.size();
        if (requestBytes > checkpoint::telemetry::MaxOtlpRequestBytes - pendingOtlpBytes)
        {
            return OtlpError(HttpServiceUnavailable, "OTLP collector routing is not configured and its pending buffer is full");
        }
        pendingOtlp.push_back(
            {.path = std::string{path}, .contentEncoding = std::string{contentEncoding}, .payload = std::string{payload}});
        pendingOtlpBytes += requestBytes;
        return {.status = HttpOk, .body = {}};
    }

    static LogTelState& Get()
    {
        SUPPRESS_WARNINGS_START
        SUPPRESS_CLANG_WARNING("-Wexit-time-destructors")
        static LogTelState Instance;
        SUPPRESS_WARNINGS_END
        return Instance;
    }
};
}    // namespace

namespace
{
class LoggingSink final : public spdlog::sinks::sink
{
    public:
    void log(spdlog::details::log_msg const& msg) override
    {
        std::string_view payload{msg.payload.data(), msg.payload.size()};
        while (!payload.empty() && (payload.back() == '\n' || payload.back() == '\r')) { payload.remove_suffix(1); }
        std::string_view   prefix;
        logs_api::Severity severity{};
        switch (msg.level)
        {
        case spdlog::level::trace:
            prefix   = "[native_ckpt:svr][TRACE] ";
            severity = logs_api::Severity::kTrace;
            break;
        case spdlog::level::debug:
            prefix   = "[native_ckpt:svr][DEBUG] ";
            severity = logs_api::Severity::kDebug;
            break;
        case spdlog::level::warn:
            prefix   = "[native_ckpt:svr][WARN] ";
            severity = logs_api::Severity::kWarn;
            break;
        case spdlog::level::err:
            prefix   = "[native_ckpt:svr][ERROR] ";
            severity = logs_api::Severity::kError;
            break;
        case spdlog::level::critical:
            prefix   = "[native_ckpt:svr][CRITICAL] ";
            severity = logs_api::Severity::kFatal;
            break;
        case spdlog::level::info:
        case spdlog::level::off:
        case spdlog::level::n_levels:
        default:
            prefix   = "[native_ckpt:svr][INFO] ";
            severity = logs_api::Severity::kInfo;
            break;
        }
        {
            std::scoped_lock lock(_mutex);
            std::string      line;
            line.reserve(prefix.size() + payload.size() + 1);
            line.append(prefix).append(payload).push_back('\n');
            static_cast<void>(std::fwrite(line.data(), sizeof(char), line.size(), stderr));
        }

        LogTelState& state = LogTelState::Get();
        if (state.enabled.load(std::memory_order_acquire))
        {
            std::scoped_lock lock(state.logs.mutex);
            if (state.logs.logger)
            {
                auto record = state.logs.logger->CreateLogRecord();
                if (record)
                {
                    record->SetSeverity(severity);
                    record->SetBody(payload);
                    state.logs.logger->EmitLogRecord(std::move(record));
                }
            }
        }
    }

    void flush() override
    {
        std::scoped_lock lock(_mutex);
        static_cast<void>(std::fflush(stderr));
    }

    void set_pattern(std::string const& /*pattern*/) override {}
    void set_formatter(std::unique_ptr<spdlog::formatter> /*formatter*/) override {}

    private:
    std::mutex _mutex;
};

}    // namespace

namespace checkpoint::logging
{
void Init()
{
    LogTelState&     state = LogTelState::Get();
    std::scoped_lock lock(state.mutex);
    if (state.logger) { return; }

    static constexpr size_t QueueSize   = 8192;
    static constexpr size_t WorkerCount = 1;
    std::call_once(state.poolOnce, [] { spdlog::init_thread_pool(QueueSize, WorkerCount); });

    // The redirected descriptor is shared with async-signal-safe crash logging.
    auto logger = std::make_shared<spdlog::async_logger>(
        "native_ckpt", std::make_shared<LoggingSink>(), spdlog::thread_pool(), spdlog::async_overflow_policy::overrun_oldest);

    std::string const         levelStr = checkpoint::config::GetConfigValue<checkpoint::config::LogLevel>();
    spdlog::level::level_enum level    = spdlog::level::from_str(levelStr);
    if (level == spdlog::level::off && levelStr != "off") { level = spdlog::level::info; }
    logger->set_level(level);
    logger->flush_on(spdlog::level::err);

    state.logger = std::move(logger);
}

void Flush()
{
    LogTelState&     state = LogTelState::Get();
    std::scoped_lock lock(state.mutex);
    if (state.logger) { state.logger->flush(); }
}

void Shutdown()
{
    LogTelState& state = LogTelState::Get();
    {
        std::scoped_lock lock(state.mutex);
        if (!state.logger) { return; }
        state.logger->flush();
        state.logger.reset();
    }
    spdlog::shutdown();
}

void LogImpl(Level level, std::string_view message)
{
    LogTelState&     state = LogTelState::Get();
    std::scoped_lock lock(state.mutex);
    if (!state.logger) { return; }
    auto toSpdlogLevel = [](Level levelIn) -> spdlog::level::level_enum {
        switch (levelIn)
        {
        case Level::Trace: return spdlog::level::trace;
        case Level::Debug: return spdlog::level::debug;
        case Level::Info: return spdlog::level::info;
        case Level::Warn: return spdlog::level::warn;
        case Level::Error: return spdlog::level::err;
        case Level::Critical: return spdlog::level::critical;
        default: break;
        }
        return spdlog::level::info;
    };

    state.logger->log(toSpdlogLevel(level), spdlog::string_view_t{message.data(), message.size()});
}

}    // namespace checkpoint::logging

namespace checkpoint::telemetry
{
MemorySampler::MemorySampler() = default;

MemorySampler::~MemorySampler()
{ Shutdown(); }

void MemorySampler::Start()
{
    if (_thread.joinable()) { return; }
    auto const seconds = checkpoint::config::GetConfigValue<checkpoint::config::MemorySampleIntervalSeconds>();
    if (seconds == 0) { return; }
    _interval = std::chrono::seconds{seconds};
    _thread   = std::jthread([this](std::stop_token stopToken) { Run_(stopToken); });
}

void MemorySampler::Shutdown()
{
    if (!_thread.joinable()) { return; }
    _thread.request_stop();
    _wake.notify_all();
    _thread.join();
}

void MemorySampler::Log(std::string_view const& trigger)
{
    if (!IsEnabled()) { return; }
    PERF() << event::ProcessMemory{
        .usage   = ProcessMemUsageInfo::Get(),
        .trigger = trigger,
    };
}

void MemorySampler::Run_(std::stop_token& stopToken)
{
    SetThreadName("nc:memsample");
    std::unique_lock lock(_waitMutex);
    while (!stopToken.stop_requested())
    {
        _wake.wait_for(lock, stopToken, _interval, [] { return false; });
        if (stopToken.stop_requested()) { break; }
        lock.unlock();
        Log("sampler");
        lock.lock();
    }
}

using UInt32Text = std::array<char, std::numeric_limits<uint32_t>::digits10 + 1>;

static std::string_view ToText(uint32_t value, UInt32Text& text)
{
    auto const result = std::to_chars(text.data(), text.data() + text.size(), value);
    return {text.data(), static_cast<size_t>(result.ptr - text.data())};
}

namespace
{

class RoutingAppender : public otlp::OtlpFileAppender
{
    public:
    RoutingAppender(std::string const& filePath, LogTelState& state LFTBND) : _state(state)
    {
        if (!filePath.empty())
        {
            _file.open(filePath, std::ios::out | std::ios::app);
            if (!_file.is_open()) { throw std::runtime_error("Failed to open file: " + filePath); }
            _file.exceptions(std::ios::badbit | std::ios::failbit);
        }
    }

    void Export(std::string_view data, std::size_t /*record_count*/) override
    {
        std::scoped_lock lock(_mutex);
        if (_file.is_open())
        {
            _file.write(data.data(), static_cast<std::streamsize>(data.size()));
            if (data.empty() || data[data.size() - 1] != '\n') { _file.put('\n'); }
            _file.flush();
        }
        std::string_view line{data.data(), data.size()};
        if (!line.empty() && line.back() == '\n') { line.remove_suffix(1); }
        if (!line.empty() && line.back() == '\r') { line.remove_suffix(1); }
        _state.Collect(line);
    }

    bool ForceFlush(std::chrono::microseconds /*timeout*/) noexcept override
    {
        std::scoped_lock lock(_mutex);
        if (_file.is_open()) { _file.flush(); }
        return true;
    }

    bool Shutdown(std::chrono::microseconds /*timeout*/) noexcept override
    {
        std::scoped_lock lock(_mutex);
        if (_file.is_open()) { _file.close(); }
        return true;
    }

    private:
    std::mutex    _mutex;
    std::ofstream _file;
    LogTelState&  _state;
};

}    // namespace

static std::shared_ptr<otlp::OtlpFileAppender>
MakeAppender(checkpoint::config::OtelConfig const& cfg, LogTelState& state, std::string_view suffix = {})
{
    bool const  writeFile = cfg.WritesProcessFile();
    std::string file      = writeFile ? cfg.ExportFile().string() : std::string{};
    if (writeFile) { file.append(suffix); }
    return std::make_shared<RoutingAppender>(file, state);
}

void Init()
{
    LogTelState&     state = LogTelState::Get();
    std::scoped_lock lock(state.mutex);
    if (state.enabled.load(std::memory_order_relaxed)) { return; }

    checkpoint::config::OtelConfig const cfg;
    if (!cfg.enabled) { return; }

    std::string const            service = cfg.service.empty() ? "nvrx-checkpoint-server" : cfg.service;
    otel_res::ResourceAttributes attrs{
        {"service.name", service},
        {"host.name", checkpoint::config::Hostname()},
        {"process.pid", static_cast<int64_t>(::getpid())},
    };
    if (!cfg.ciPhase.empty()) { attrs.SetAttribute("nvrx.ci_phase", cfg.ciPhase); }
    if (!cfg.runId.empty()) { attrs.SetAttribute("nemo.run.id", cfg.runId); }
    otel_res::Resource resource = otel_res::Resource::Create(attrs);

    otlp::OtlpFileExporterOptions spanOpts;
    spanOpts.backend_options                          = MakeAppender(cfg, state);
    auto                                 spanExporter = otlp::OtlpFileExporterFactory::Create(spanOpts);
    trace_sdk::BatchSpanProcessorOptions bspOpts;
    static constexpr auto                ExportInterval = std::chrono::milliseconds{1000};
    static constexpr auto                ExportTimeout  = std::chrono::milliseconds{500};
    bspOpts.schedule_delay_millis                       = ExportInterval;
    auto processor                                      = trace_sdk::BatchSpanProcessorFactory::Create(std::move(spanExporter), bspOpts);
    state.traceProvider                                 = trace_sdk::TracerProviderFactory::Create(std::move(processor), resource);
    trace_api::Provider::SetTracerProvider(state.traceProvider);
    state.tracer = state.traceProvider->GetTracer("nvrx.checkpoint");

    if (cfg.metrics)
    {
        otlp::OtlpFileMetricExporterOptions exporterOpts;
        exporterOpts.backend_options                                     = MakeAppender(cfg, state, ".metrics");
        auto                                              metricExporter = otlp::OtlpFileMetricExporterFactory::Create(exporterOpts);
        metrics_sdk::PeriodicExportingMetricReaderOptions readerOpts;
        readerOpts.export_interval_millis = ExportInterval;
        readerOpts.export_timeout_millis  = ExportTimeout;
        auto reader         = metrics_sdk::PeriodicExportingMetricReaderFactory::Create(std::move(metricExporter), readerOpts);
        auto uMeterProvider = metrics_sdk::MeterProviderFactory::Create(std::make_unique<metrics_sdk::ViewRegistry>(), resource);
        uMeterProvider->AddMetricReader(std::move(reader));
        state.metrics.provider = std::move(uMeterProvider);
        metrics_api::Provider::SetMeterProvider(state.metrics.provider);
        std::scoped_lock metricLock(state.metrics.mutex);
        state.metrics.meter = state.metrics.provider->GetMeter("nvrx.checkpoint");
    }

    if (cfg.logs)
    {
        otlp::OtlpFileLogRecordExporterOptions exporterOpts;
        exporterOpts.backend_options                         = MakeAppender(cfg, state, ".logs");
        auto                                     logExporter = otlp::OtlpFileLogRecordExporterFactory::Create(exporterOpts);
        logs_sdk::BatchLogRecordProcessorOptions blpOpts;
        blpOpts.schedule_delay_millis = ExportInterval;
        auto logProcessor             = logs_sdk::BatchLogRecordProcessorFactory::Create(std::move(logExporter), blpOpts);
        state.logs.provider           = logs_sdk::LoggerProviderFactory::Create(std::move(logProcessor), resource);
        logs_api::Provider::SetLoggerProvider(state.logs.provider);
        std::scoped_lock logLock(state.logs.mutex);
        state.logs.logger = state.logs.provider->GetLogger("nvrx.checkpoint", "nvrx.checkpoint");
    }

    std::call_once(state.shutdownOnce, [] { std::atexit(+[] { Shutdown(); }); });

    state.enabled.store(true, std::memory_order_release);
    state.sampler.Start();
}

bool IsEnabled() noexcept
{ return LogTelState::Get().enabled.load(std::memory_order_acquire); }

void Shutdown()
{
    LogTelState& state = LogTelState::Get();
    state.sampler.Shutdown();
    {
        std::scoped_lock lock(state.mutex);
        if (state.enabled.exchange(false, std::memory_order_acq_rel))
        {
            if (state.traceProvider) { state.traceProvider->ForceFlush(); }
            {
                std::scoped_lock exportLock(state.metrics.mutex, state.logs.mutex);
                if (state.metrics.provider) { state.metrics.provider->ForceFlush(); }
                if (state.logs.provider) { state.logs.provider->ForceFlush(); }
                state.metrics.histograms.clear();
                state.metrics.meter.reset();
                state.logs.logger.reset();
                state.lastId = {};
            }

            state.saveStarts.entries.clear();
            state.activeSpans.entries.clear();
            state.tracer.reset();

            trace_api::Provider::SetTracerProvider(std::make_shared<trace_api::NoopTracerProvider>());
            logs_api::Provider::SetLoggerProvider(std::make_shared<logs_api::NoopLoggerProvider>());
            metrics_api::Provider::SetMeterProvider(std::make_shared<metrics_api::NoopMeterProvider>());
            state.traceProvider.reset();
            state.metrics.provider.reset();
            state.logs.provider.reset();
        }
    }

    std::scoped_lock collectorLock(state.collectorMutex);
    state.collector.emplace<std::monostate>();
    state.pendingOtlp.clear();
    state.pendingOtlpBytes = 0;
}

template <typename Labels> static void RecordHistogramImpl(std::string_view name, double value, std::string_view unit, Labels const& labels)
{
    LogTelState::MetricState& metrics = LogTelState::Get().metrics;
    std::scoped_lock          lock(metrics.mutex);
    if (!metrics.meter) { return; }
    auto it = metrics.histograms.find(name);
    if (it == metrics.histograms.end())
    {
        it = metrics.histograms.emplace(std::string{name}, metrics.meter->CreateDoubleHistogram(name, "nvrx checkpoint metric", unit))
                 .first;
    }
    if (it->second) { it->second->Record(value, otel_common::KeyValueIterableView<Labels>{labels}, opentelemetry::context::Context{}); }
}
void InitLocalCollector(std::filesystem::path const& filePath)
{
    if (filePath.empty()) { return; }
    LogTelState&     state = LogTelState::Get();
    std::scoped_lock lock(state.collectorMutex);
    state.InitCollector<FileCollector>(filePath);
}

void InitRelayCollector(std::string_view const& relayToRemote)
{
    if (relayToRemote.empty()) { return; }
    LogTelState&     state = LogTelState::Get();
    std::scoped_lock lock(state.collectorMutex);
    state.InitCollector<RelayCollector>(relayToRemote);
}

void Collect(std::string_view const& payload)
{ LogTelState::Get().Collect(payload); }

OtlpResponse ReceiveOtlp(std::string_view const& path,
                         std::string_view const& contentType,
                         std::string_view const& contentEncoding,
                         std::string_view const& payload)
{ return LogTelState::Get().ReceiveOtlp(path, contentType, contentEncoding, payload); }

static void EmitProcessMemory(LogTelState& state, std::scoped_lock<std::mutex> const& /* lock */, event::ProcessMemory const& fields)
{
    if (!state.logs.logger) { return; }
    auto record = state.logs.logger->CreateLogRecord();
    record->SetTimestamp(std::chrono::system_clock::now());
    record->SetEventId(0, event::ProcessMemory::Name);
    record->SetAttribute("nvrx.memory.rss_bytes", static_cast<int64_t>(fields.usage.rssBytes));
    record->SetAttribute("nvrx.memory.peak_rss_bytes", static_cast<int64_t>(fields.usage.peakRssBytes));
    record->SetAttribute("nvrx.memory.anonymous_rss_bytes", static_cast<int64_t>(fields.usage.anonymousRssBytes));
    record->SetAttribute("nvrx.memory.file_rss_bytes", static_cast<int64_t>(fields.usage.fileRssBytes));
    record->SetAttribute("nvrx.memory.shmem_rss_bytes", static_cast<int64_t>(fields.usage.shmemRssBytes));
    record->SetAttribute("nvrx.memory.locked_bytes", static_cast<int64_t>(fields.usage.lockedBytes));
    record->SetAttribute("nvrx.memory.pinned_bytes", static_cast<int64_t>(fields.usage.pinnedBytes));
    record->SetAttribute("nvrx.memory.swap_bytes", static_cast<int64_t>(fields.usage.swapBytes));
    record->SetAttribute("nvrx.memory.trigger", fields.trigger);
    record->SetAttribute("nvrx.session_id", static_cast<int64_t>(static_cast<uint32_t>(state.lastId.sessionId)));
    record->SetAttribute("nvrx.rank", static_cast<int64_t>(static_cast<uint32_t>(state.lastId.rank)));
    record->SetAttribute("nvrx.snapshot_id", static_cast<int64_t>(static_cast<uint32_t>(state.lastId.snapshotId)));
    state.logs.logger->EmitLogRecord(std::move(record));
}

void Emitter<event::ProcessMemory>::Emit(event::ProcessMemory const& fields)
{
    LogTelState& state = LogTelState::Get();
    if (!state.enabled.load(std::memory_order_acquire)) { return; }

    std::scoped_lock<std::mutex> lock(state.mutex);
    if (!state.enabled.load(std::memory_order_relaxed)) { return; }
    EmitProcessMemory(state, lock, fields);
}

void Emitter<span::CkptSave::Begin>::Emit(span::CkptSave::Begin const& fields)
{
    fmt::print(stderr,
               "[native_ckpt:svr][{}][PERF] Rank={} Begin span={} started_at_ns={} snapshot={}\n",
               static_cast<uint32_t>(fields.id.sessionId),
               static_cast<uint32_t>(fields.id.rank),
               span::CkptSave::Name,
               fields.startedAt.count(),
               static_cast<uint32_t>(fields.id.snapshotId));

    LogTelState& state = LogTelState::Get();
    if (!state.enabled.load(std::memory_order_acquire)) { return; }

    std::scoped_lock<std::mutex> lock(state.mutex);
    if (!state.enabled.load(std::memory_order_relaxed)) { return; }
    state.lastId = fields.id;
    EmitProcessMemory(state, lock, {.usage = ProcessMemUsageInfo::Get(), .trigger = span::CkptSave::Name});
    state.saveStarts.entries.push_back({.id = fields.id, .startedAt = fields.startedAt});
}

void Emitter<span::CkptSave::End>::Emit(span::CkptSave::End const& fields)
{
    auto const writeSeconds = static_cast<double>(fields.d2hDuration.count()) / 1e6;    // NOLINT
    auto const bwMbps
        = writeSeconds == 0.0 ? 0.0 : static_cast<double>(fields.totalTensorBytes) / (1024.0 * 1024.0) / writeSeconds;    // NOLINT
    fmt::print(stderr,
               "[native_ckpt:svr][{}][PERF] Rank={} Finished bw_mbps={:.2f} "
               "| Time(us):  wr_tnsr_us={} snapshot_us={} total_us={} "
               "| Count: num_tensors={} total_d2h={} "
               "| Bytes: total_written_bytes={} avg_d2h_rq={} avg_tnsr={} snapshot={}\n",
               static_cast<uint32_t>(fields.id.sessionId),
               static_cast<uint32_t>(fields.id.rank),
               bwMbps,
               fields.d2hDuration,
               fields.snapshotDuration,
               fields.totalDuration,
               fields.numTensors,
               fields.d2hRequestCount,
               fields.totalWrittenBytes,
               fields.totalTensorBytes / std::max(fields.d2hRequestCount, uint64_t{1}),
               fields.totalTensorBytes / std::max(fields.numTensors, uint64_t{1}),
               static_cast<uint32_t>(fields.id.snapshotId));

    LogTelState& state = LogTelState::Get();
    if (!state.enabled.load(std::memory_order_acquire)) { return; }

    auto const endSystem   = std::chrono::system_clock::now();
    auto const endSteady   = std::chrono::steady_clock::now();
    auto       startSystem = endSystem - fields.totalDuration;
    auto const startSteady = endSteady - std::chrono::duration_cast<std::chrono::steady_clock::duration>(fields.totalDuration);
    {
        std::scoped_lock<std::mutex> lock(state.mutex);
        if (!state.enabled.load(std::memory_order_relaxed) || !state.tracer) { return; }

        auto const entry = std::ranges::find_if(state.saveStarts.entries, [&fields](auto const& start) { return start.id == fields.id; });
        if (entry != state.saveStarts.entries.end())
        {
            startSystem
                = std::chrono::system_clock::time_point{std::chrono::duration_cast<std::chrono::system_clock::duration>(entry->startedAt)};
            state.saveStarts.entries.erase(entry);
        }
        state.lastId = fields.id;
        EmitProcessMemory(state, lock, {.usage = ProcessMemUsageInfo::Get(), .trigger = span::CkptSave::Name});

        trace_api::StartSpanOptions startOptions;
        startOptions.start_system_time = startSystem;
        startOptions.start_steady_time = startSteady;
        auto ckptSpan                  = state.tracer->StartSpan(span::CkptSave::Name, startOptions);
        ckptSpan->SetAttribute("nvrx.session_id", static_cast<int64_t>(static_cast<uint32_t>(fields.id.sessionId)));
        ckptSpan->SetAttribute("nvrx.rank", static_cast<int64_t>(static_cast<uint32_t>(fields.id.rank)));
        ckptSpan->SetAttribute("nvrx.snapshot_id", static_cast<int64_t>(static_cast<uint32_t>(fields.id.snapshotId)));
        ckptSpan->SetAttribute("nvrx.bw_mbps", bwMbps);
        ckptSpan->SetAttribute("nvrx.wr_tnsr_us", static_cast<int64_t>(fields.d2hDuration.count()));
        ckptSpan->SetAttribute("nvrx.snapshot_us", static_cast<int64_t>(fields.snapshotDuration.count()));
        ckptSpan->SetAttribute("nvrx.total_us", static_cast<int64_t>(fields.totalDuration.count()));
        ckptSpan->SetAttribute("nvrx.num_tensors", static_cast<int64_t>(fields.numTensors));
        ckptSpan->SetAttribute("nvrx.total_d2h", static_cast<int64_t>(fields.d2hRequestCount));
        ckptSpan->SetAttribute("nvrx.total_tensor_bytes", static_cast<int64_t>(fields.totalTensorBytes));
        ckptSpan->SetAttribute("nvrx.total_written_bytes", static_cast<int64_t>(fields.totalWrittenBytes));
        ckptSpan->SetAttribute("nvrx.avg_d2h_rq",
                               static_cast<int64_t>(fields.totalTensorBytes / std::max(fields.d2hRequestCount, uint64_t{1})));
        ckptSpan->SetAttribute("nvrx.avg_tnsr", static_cast<int64_t>(fields.totalTensorBytes / std::max(fields.numTensors, uint64_t{1})));
        trace_api::EndSpanOptions endOptions;
        endOptions.end_steady_time = endSteady;
        ckptSpan->End(endOptions);
    }

    using Label = std::pair<std::string_view, otel_common::AttributeValue>;
    UInt32Text                 sessionText;
    UInt32Text                 rankText;
    UInt32Text                 snapshotText;
    std::array<Label, 3> const dimensions{
        Label{"nvrx.session_id", ToText(static_cast<uint32_t>(fields.id.sessionId), sessionText)},
        Label{"nvrx.rank", ToText(static_cast<uint32_t>(fields.id.rank), rankText)},
        Label{"nvrx.snapshot_id", ToText(static_cast<uint32_t>(fields.id.snapshotId), snapshotText)},
    };
    RecordHistogramImpl("nvrx.ckpt.save.bw_mbps", bwMbps, "MB/s", dimensions);
    RecordHistogramImpl("nvrx.ckpt.save.wr_tnsr_us", static_cast<double>(fields.d2hDuration.count()), "us", dimensions);
    RecordHistogramImpl("nvrx.ckpt.save.total_written_bytes", static_cast<double>(fields.totalWrittenBytes), "By", dimensions);
}

void Emitter<span::SvrFinalize::Begin>::Emit(span::SvrFinalize::Begin const& fields)
{
    LogTelState& state = LogTelState::Get();
    if (!state.enabled.load(std::memory_order_acquire)) { return; }

    std::scoped_lock<std::mutex> lock(state.mutex);
    if (!state.enabled.load(std::memory_order_relaxed) || !state.tracer) { return; }

    state.lastId = fields.id;
    EmitProcessMemory(state, lock, {.usage = ProcessMemUsageInfo::Get(), .trigger = span::SvrFinalize::Name});

    auto finalizeSpan = state.tracer->StartSpan(span::SvrFinalize::Name);
    finalizeSpan->SetAttribute("nvrx.session_id", static_cast<int64_t>(static_cast<uint32_t>(fields.id.sessionId)));
    finalizeSpan->SetAttribute("nvrx.rank", static_cast<int64_t>(static_cast<uint32_t>(fields.id.rank)));
    finalizeSpan->SetAttribute("nvrx.snapshot_id", static_cast<int64_t>(static_cast<uint32_t>(fields.id.snapshotId)));
    state.activeSpans.entries.push_back(
        {.id = {.kind = LogTelState::ActiveSpanKind::SvrFinalize, .id = fields.id}, .span = std::move(finalizeSpan)});
}

void Emitter<span::SvrFinalize::End>::Emit(span::SvrFinalize::End const& fields)
{
    LogTelState& state = LogTelState::Get();
    if (!state.enabled.load(std::memory_order_acquire)) { return; }

    std::scoped_lock<std::mutex> lock(state.mutex);
    if (!state.enabled.load(std::memory_order_relaxed)) { return; }

    state.lastId = fields.id;
    EmitProcessMemory(state, lock, {.usage = ProcessMemUsageInfo::Get(), .trigger = span::SvrFinalize::Name});

    auto const id    = LogTelState::ActiveSpanId{.kind = LogTelState::ActiveSpanKind::SvrFinalize, .id = fields.id};
    auto const entry = std::ranges::find_if(state.activeSpans.entries, [&id](auto const& active) { return active.id == id; });
    if (entry == state.activeSpans.entries.end()) { return; }
    auto finalizeSpan = std::move(entry->span);
    state.activeSpans.entries.erase(entry);
    finalizeSpan->End();
}

void Emitter<span::SvrWriteMetadata::Begin>::Emit(span::SvrWriteMetadata::Begin const& fields)
{
    LogTelState& state = LogTelState::Get();
    if (!state.enabled.load(std::memory_order_acquire)) { return; }

    std::scoped_lock<std::mutex> lock(state.mutex);
    if (!state.enabled.load(std::memory_order_relaxed) || !state.tracer) { return; }

    state.lastId = fields.id;
    EmitProcessMemory(state, lock, {.usage = ProcessMemUsageInfo::Get(), .trigger = span::SvrWriteMetadata::Name});

    auto metadataSpan = state.tracer->StartSpan(span::SvrWriteMetadata::Name);
    metadataSpan->SetAttribute("nvrx.session_id", static_cast<int64_t>(static_cast<uint32_t>(fields.id.sessionId)));
    metadataSpan->SetAttribute("nvrx.snapshot_id", static_cast<int64_t>(static_cast<uint32_t>(fields.id.snapshotId)));
    state.activeSpans.entries.push_back(
        {.id = {.kind = LogTelState::ActiveSpanKind::SvrWriteMetadata, .id = fields.id}, .span = std::move(metadataSpan)});
}

void Emitter<span::SvrWriteMetadata::End>::Emit(span::SvrWriteMetadata::End const& fields)
{
    LogTelState& state = LogTelState::Get();
    if (!state.enabled.load(std::memory_order_acquire)) { return; }

    std::scoped_lock<std::mutex> lock(state.mutex);
    if (!state.enabled.load(std::memory_order_relaxed)) { return; }

    state.lastId = fields.id;
    EmitProcessMemory(state, lock, {.usage = ProcessMemUsageInfo::Get(), .trigger = span::SvrWriteMetadata::Name});

    auto const id    = LogTelState::ActiveSpanId{.kind = LogTelState::ActiveSpanKind::SvrWriteMetadata, .id = fields.id};
    auto const entry = std::ranges::find_if(state.activeSpans.entries, [&id](auto const& active) { return active.id == id; });
    if (entry == state.activeSpans.entries.end()) { return; }
    auto metadataSpan = std::move(entry->span);
    state.activeSpans.entries.erase(entry);
    metadataSpan->End();
}

void Emitter<span::SvrDurable::Begin>::Emit(span::SvrDurable::Begin const& fields)
{

    LogTelState& state = LogTelState::Get();
    if (!state.enabled.load(std::memory_order_acquire)) { return; }
    std::scoped_lock<std::mutex> lock(state.mutex);
    if (!state.enabled.load(std::memory_order_relaxed) || !state.tracer) { return; }

    state.lastId = fields.id;
    EmitProcessMemory(state, lock, {.usage = ProcessMemUsageInfo::Get(), .trigger = span::SvrDurable::Name});

    auto const                  nowSystem = std::chrono::system_clock::now();
    auto const                  nowSteady = std::chrono::steady_clock::now();
    trace_api::StartSpanOptions options;
    options.start_system_time = fields.startedAt;
    options.start_steady_time = nowSteady - std::chrono::duration_cast<std::chrono::steady_clock::duration>(nowSystem - fields.startedAt);
    auto durableSpan          = state.tracer->StartSpan(span::SvrDurable::Name, options);
    durableSpan->SetAttribute("nvrx.session_id", static_cast<int64_t>(static_cast<uint32_t>(fields.id.sessionId)));
    durableSpan->SetAttribute("nvrx.rank", static_cast<int64_t>(static_cast<uint32_t>(fields.id.rank)));
    durableSpan->SetAttribute("nvrx.snapshot_id", static_cast<int64_t>(static_cast<uint32_t>(fields.id.snapshotId)));
    state.activeSpans.entries.push_back(
        {.id = {.kind = LogTelState::ActiveSpanKind::SvrDurable, .id = fields.id}, .span = std::move(durableSpan)});
}

void Emitter<span::SvrDurable::End>::Emit(span::SvrDurable::End const& fields)
{
    LogTelState& state = LogTelState::Get();
    if (!state.enabled.load(std::memory_order_acquire)) { return; }

    std::scoped_lock<std::mutex> lock(state.mutex);
    if (!state.enabled.load(std::memory_order_relaxed)) { return; }
    state.lastId = fields.id;
    EmitProcessMemory(state, lock, {.usage = ProcessMemUsageInfo::Get(), .trigger = span::SvrDurable::Name});

    auto const id    = LogTelState::ActiveSpanId{.kind = LogTelState::ActiveSpanKind::SvrDurable, .id = fields.id};
    auto const entry = std::ranges::find_if(state.activeSpans.entries, [&id](auto const& active) { return active.id == id; });
    if (entry == state.activeSpans.entries.end()) { return; }
    auto durableSpan = std::move(entry->span);
    state.activeSpans.entries.erase(entry);
    durableSpan->End();
}

}    // namespace checkpoint::telemetry
