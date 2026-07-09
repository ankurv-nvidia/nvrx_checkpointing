// ckpt_stress_test — scale the checkpoint web service's connection layer.
//
// At large training scale (thousands of nodes) every non-coordinator node's
// checkpoint server opens a *fresh* TCP connection to the coordinator (peers[0],
// "node:0") for each notify/data message — finalizecomplete, writeckptcomplete,
// shardedstateregistry. node:0 services those with a single accept coroutine that
// runs each request handler synchronously before accepting the next connection, so
// accepts are serialized and pending connections queue in the kernel listen backlog
// (SOMAXCONN). When the offered connection rate outruns the serial drain and the
// backlog fills, new connects stall, reset, or are refused — the "connection refused
// on node:0" reported at 256+ GPUs.
//
// This driver stands up the real web service in-process (RunInprocServer) and then
// emulates up to `--max-nodes` peer nodes hammering it with the exact connect-per-
// message pattern, ramping concurrency and reporting, per level: connections
// attempted / succeeded / refused / reset / timed-out / addr-unavailable, connect+RTT
// latency percentiles, achieved connections-per-second, and the kernel accept-queue
// overflow delta. It flags the first concurrency level at which connections stop
// succeeding cleanly and attributes the dominant failure mode (server-side backlog
// stall/RST vs client-side ephemeral-port exhaustion). `--saves N` replays the ramp N
// times to model steady-state training and surface cumulative effects. `--external`
// drives an already-running server (a real checkpoint_server or remote host).

#include "CommonMacros.h"
#include "messages.h"
#include "server.h"

#include <boost/asio/co_spawn.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/strand.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/program_options.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <sys/resource.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace bpo   = boost::program_options;
namespace net   = boost::asio;
namespace beast = boost::beast;
namespace http  = boost::beast::http;
using tcp       = boost::asio::ip::tcp;
using Clock     = std::chrono::steady_clock;

namespace
{

double ToMs(Clock::duration d)
{ return std::chrono::duration<double, std::milli>(d).count(); }

// Endpoint the storm targets. Both faithfully reproduce the connect-per-message
// pattern; the difference is only how much work the handler does per request.
enum class Endpoint : uint8_t
{
    // GET /api/ping — cheapest handler; isolates the accept/backlog scaling axis.
    Ping,
    // POST /api/peer/shardedstateregistry/{sess}/{rank} with an empty body — the
    // real "send data" path (header + body read + pending-map lock). Needs no live
    // session: an unknown session parks the (empty) fragment list in pending data.
    ShardedState,
};

enum class Outcome : uint8_t
{
    Ok,
    HttpError,      // connected + exchanged, but status != 200
    ConnRefused,    // ECONNREFUSED — node:0 RST'd the SYN (full accept queue w/ abort-on-overflow, or not listening)
    ConnReset,      // peer reset an established/half-open connection
    Timeout,        // connect or exchange exceeded the deadline (backlog stall)
    AddrUnavail,    // EADDRNOTAVAIL / EADDRINUSE — client ephemeral-port exhaustion (connect-per-message hazard)
    ResolveErr,
    OtherErr,
    Count_,
};

Outcome Classify(boost::system::error_code const& ec)
{
    if (ec == beast::error::timeout || ec == net::error::timed_out) { return Outcome::Timeout; }
    if (ec == net::error::connection_refused) { return Outcome::ConnRefused; }
    if (ec == net::error::address_in_use || ec == net::error::address_family_not_supported
        || ec == boost::system::errc::address_not_available)
    {
        return Outcome::AddrUnavail;
    }
    if (ec == net::error::connection_reset || ec == net::error::broken_pipe || ec == net::error::eof) { return Outcome::ConnReset; }
    if (ec == net::error::connection_aborted) { return Outcome::ConnReset; }
    if (ec == net::error::host_unreachable || ec == net::error::host_not_found) { return Outcome::ResolveErr; }
    return Outcome::OtherErr;
}

struct Config
{
    std::string               host{"127.0.0.1"};
    uint16_t                  port{checkpoint::ipc::Message::Port};
    Endpoint                  endpoint{Endpoint::Ping};
    uint32_t                  iters{3};              // messages each node sends per save (request waves per level)
    uint32_t                  saves{1};              // checkpoint saves to replay the ramp over (steady-state modeling)
    uint32_t                  maxInflight{20480};    // cap on simultaneous connections (single-host driver ceiling)
    std::chrono::milliseconds connectTimeout{5000};
    std::string               hostHeader;    // filled in main
    std::string               target;        // filled per request for ShardedState
};

struct Stats
{
    std::array<std::atomic<long>, static_cast<size_t>(Outcome::Count_)> counts{};

    std::mutex          latMu;
    std::vector<double> connectMs;    // successful connects only
    std::vector<double> rttMs;        // successful full exchanges only

    void Record(Outcome o, double connect, double rtt, bool connected, bool exchanged)
    {
        counts[static_cast<size_t>(o)].fetch_add(1, std::memory_order_relaxed);
        if (connected || exchanged)
        {
            std::scoped_lock lock(latMu);
            if (connected) { connectMs.push_back(connect); }
            if (exchanged) { rttMs.push_back(rtt); }
        }
    }

    long Get(Outcome o) const { return counts[static_cast<size_t>(o)].load(std::memory_order_relaxed); }
    long Total() const
    {
        long t = 0;
        for (auto const& c : counts) { t += c.load(std::memory_order_relaxed); }
        return t;
    }
};

double Percentile(std::vector<double>& v, double p)
{
    if (v.empty()) { return 0.0; }
    std::sort(v.begin(), v.end());
    if (p <= 0.0) { return v.front(); }
    // p > 0 so ceil(...) >= 1; the -1 converts the 1-based rank to a 0-based index.
    auto idx = static_cast<size_t>(std::ceil(p / 100.0 * static_cast<double>(v.size()))) - 1;
    idx      = std::min(idx, v.size() - 1);
    return v[idx];
}

// One faithful peer->coordinator round trip: fresh connect, one HTTP request, read
// the response, close. Records the outcome and per-phase latency into `stats`.
net::awaitable<void> DoOneRequest(tcp::resolver::results_type const& eps, Config const& cfg, Stats& stats)
{
    auto              ex = co_await net::this_coro::executor;
    beast::tcp_stream stream(ex);
    stream.expires_after(cfg.connectTimeout);

    auto const t0        = Clock::now();
    double     connectMs = 0.0;
    double     rttMs     = 0.0;
    bool       connected = false;
    bool       exchanged = false;
    Outcome    outcome   = Outcome::Ok;

    try
    {
        co_await stream.async_connect(eps, net::use_awaitable);
        connectMs = ToMs(Clock::now() - t0);
        connected = true;

        http::request<http::empty_body> req;
        req.version(11);
        req.set(http::field::host, cfg.hostHeader);
        req.keep_alive(false);
        if (cfg.endpoint == Endpoint::Ping)
        {
            req.method(http::verb::get);
            req.target("/api/ping");
        }
        else
        {
            req.method(http::verb::post);
            req.target(cfg.target);
            req.set(http::field::content_type, "application/octet-stream");
            req.content_length(0);
        }

        co_await http::async_write(stream, req, net::use_awaitable);

        beast::flat_buffer                buffer;
        http::response<http::string_body> res;
        co_await http::async_read(stream, buffer, res, net::use_awaitable);
        rttMs     = ToMs(Clock::now() - t0);
        exchanged = true;
        outcome   = (res.result() == http::status::ok) ? Outcome::Ok : Outcome::HttpError;

        beast::error_code ec;
        stream.socket().shutdown(tcp::socket::shutdown_both, ec);
    } catch (boost::system::system_error const& e) { outcome = Classify(e.code()); } catch (...)
    {
        outcome = Outcome::OtherErr;
    }

    stats.Record(outcome, connectMs, rttMs, connected, exchanged);
    co_return;
}

struct LevelResult
{
    uint32_t nodes{};
    long     attempted{};
    long     ok{};
    long     refused{};
    long     reset{};
    long     timeout{};
    long     addrUnavail{};
    long     otherErr{};
    double   connectP50{};
    double   connectP99{};
    double   connectMax{};
    double   rttP50{};
    double   rttP99{};
    double   wallMs{};
    double   connPerSec{};
};

// Fires `nodes * iters` requests, holding up to `min(nodes, maxInflight)` in flight
// at once (a synchronized thundering herd — all peers finalize a checkpoint at nearly
// the same instant). Each completion spawns a replacement until the level's request
// budget is exhausted. maxInflight bounds the single-host driver so it stays stable
// while `nodes` still sets the total offered load.
LevelResult RunLevel(net::io_context& ioc, tcp::resolver::results_type const& eps, Config const& cfg, uint32_t nodes)
{
    Stats             stats;
    uint32_t const    concurrency = std::min(nodes, cfg.maxInflight);
    long const        total       = static_cast<long>(nodes) * cfg.iters;
    std::atomic<long> remaining{total};
    std::atomic<long> inFlight{0};
    std::atomic<long> completed{0};

    std::mutex              doneMu;
    std::condition_variable doneCv;

    auto const start = Clock::now();

    // Recursive pump: keep up to `nodes` requests in flight. Spawns run on the io
    // threads; atomics gate the concurrency and the per-level budget.
    auto pump = [&](auto&& self) -> void {
        for (;;)
        {
            if (inFlight.load(std::memory_order_relaxed) >= static_cast<long>(concurrency)) { break; }
            long r = remaining.load(std::memory_order_relaxed);
            if (r <= 0) { break; }
            if (!remaining.compare_exchange_weak(r, r - 1, std::memory_order_relaxed)) { continue; }

            inFlight.fetch_add(1, std::memory_order_relaxed);
            // Each request runs on its own strand: beast::tcp_stream is not
            // thread-safe, and its timeout timer runs concurrently with the socket
            // op, so without a strand the two race across io threads and corrupt the
            // reactor during teardown of a timed-out connection.
            net::co_spawn(net::make_strand(ioc), DoOneRequest(eps, cfg, stats), [&, self](std::exception_ptr) {
                inFlight.fetch_sub(1, std::memory_order_relaxed);
                long const c = completed.fetch_add(1, std::memory_order_relaxed) + 1;
                if (c >= total)
                {
                    {
                        std::scoped_lock lock(doneMu);
                    }
                    doneCv.notify_all();
                }
                else
                {
                    self(self);
                }
            });
        }
    };

    pump(pump);

    {
        std::unique_lock lock(doneMu);
        doneCv.wait(lock, [&] { return completed.load(std::memory_order_relaxed) >= total; });
    }

    double const wallMs = ToMs(Clock::now() - start);

    LevelResult lr;
    lr.nodes       = nodes;
    lr.attempted   = stats.Total();
    lr.ok          = stats.Get(Outcome::Ok);
    lr.refused     = stats.Get(Outcome::ConnRefused);
    lr.reset       = stats.Get(Outcome::ConnReset);
    lr.timeout     = stats.Get(Outcome::Timeout);
    lr.addrUnavail = stats.Get(Outcome::AddrUnavail);
    lr.otherErr    = stats.Get(Outcome::HttpError) + stats.Get(Outcome::ResolveErr) + stats.Get(Outcome::OtherErr);
    lr.connectP50  = Percentile(stats.connectMs, 50);
    lr.connectP99  = Percentile(stats.connectMs, 99);
    lr.connectMax  = stats.connectMs.empty() ? 0.0 : *std::max_element(stats.connectMs.begin(), stats.connectMs.end());
    lr.rttP50      = Percentile(stats.rttMs, 50);
    lr.rttP99      = Percentile(stats.rttMs, 99);
    lr.wallMs      = wallMs;
    lr.connPerSec  = wallMs > 0 ? (static_cast<double>(lr.ok) / (wallMs / 1000.0)) : 0.0;
    return lr;
}

long ReadSysctl(char const* path)
{
    std::ifstream f(path);
    long          v = -1;
    if (f) { f >> v; }
    return v;
}

// Reads a named TcpExt counter (e.g. "ListenOverflows") from /proc/net/netstat.
// ListenOverflows increments each time a completed connection cannot be queued
// because the listen socket's accept backlog is full — the kernel-level fingerprint
// of node:0 failing to accept fast enough. Returns -1 if unavailable.
long ReadTcpExt(std::string_view counter)
{
    std::ifstream f("/proc/net/netstat");
    std::string   header;
    std::string   values;
    while (std::getline(f, header))
    {
        if (header.rfind("TcpExt:", 0) != 0) { continue; }
        if (!std::getline(f, values)) { break; }
        std::istringstream hs(header);
        std::istringstream vs(values);
        std::string        hk;
        std::string        vk;
        hs >> hk;    // "TcpExt:"
        vs >> vk;
        std::string name;
        std::string val;
        while (hs >> name && vs >> val)
        {
            if (name == counter) { return std::stol(val); }
        }
    }
    return -1;
}

void RaiseFdLimit()
{
    rlimit rl{};
    if (getrlimit(RLIMIT_NOFILE, &rl) == 0)
    {
        rl.rlim_cur = rl.rlim_max;
        setrlimit(RLIMIT_NOFILE, &rl);
    }
}

// Unlink the in-process server's shm request queues. Called at startup to clear
// segments a previously force-exited run left behind (each is ~1 GiB), and on the
// forced-exit path below.
void CleanStaleShm()
{
    for (uint8_t r = 0; r < checkpoint::ipc::MaxRanksPerNode; ++r)
    {
        auto name = checkpoint::ipc::Message::RequestChannelName(0, r);
        boost::interprocess::shared_memory_object::remove(name.c_str());
    }
}

}    // namespace

int main(int argc, char** argv)
try
{
    Config      cfg;
    uint32_t    maxNodes     = 6144;
    uint32_t    singleNodes  = 0;
    std::string endpointStr  = "ping";
    bool        external     = false;
    bool        failOnErrors = false;
    std::string rampStr;
    unsigned    ioThreads = std::min(16U, std::max(2U, std::thread::hardware_concurrency()));

    bpo::options_description desc("ckpt_stress_test — scale the checkpoint web service connection layer");
    desc.add_options()                                                                                                           //
        ("help", "produce help message")                                                                                         //
        ("max-nodes", bpo::value<uint32_t>(&maxNodes), "top of the concurrency ramp (simulated peer nodes); default 6144")       //
        ("nodes", bpo::value<uint32_t>(&singleNodes), "run a single level at this concurrency instead of the ramp")              //
        ("ramp", bpo::value<std::string>(&rampStr), "comma-separated concurrency levels, e.g. 64,256,1024,4096,6144")            //
        ("iters", bpo::value<uint32_t>(&cfg.iters), "messages each node sends per save (waves per level); default 3")            //
        ("saves", bpo::value<uint32_t>(&cfg.saves), "checkpoint saves to replay the ramp over; default 1 (steady-state: >1)")    //
        ("max-inflight", bpo::value<uint32_t>(&cfg.maxInflight), "cap on simultaneous connections; default 20480")               //
        ("endpoint", bpo::value<std::string>(&endpointStr), "ping | shardedstate; default ping")                                 //
        ("host", bpo::value<std::string>(&cfg.host), "coordinator host; default 127.0.0.1")                                      //
        ("port", bpo::value<uint16_t>(&cfg.port), "coordinator port; default 41000")                                             //
        ("connect-timeout-ms", bpo::value<uint32_t>()->default_value(5000), "per-request deadline in ms")                        //
        ("io-threads", bpo::value<unsigned>(&ioThreads), "client io_context worker threads")                                     //
        ("external", bpo::bool_switch(&external), "do not start an in-process server; stress a running one at --host/--port")    //
        ("fail-on-errors", bpo::bool_switch(&failOnErrors), "exit 2 if any hard connection failure occurs (CI gate)");           //

    bpo::variables_map vm;
    bpo::store(bpo::parse_command_line(argc, argv, desc), vm);
    bpo::notify(vm);

    if (vm.contains("help"))
    {
        std::cerr << desc << '\n';
        return 0;
    }

    if (endpointStr != "ping" && endpointStr != "shardedstate")
    {
        throw std::invalid_argument("--endpoint must be 'ping' or 'shardedstate'");
    }
    cfg.connectTimeout = std::chrono::milliseconds(vm["connect-timeout-ms"].as<uint32_t>());
    cfg.endpoint       = (endpointStr == "shardedstate") ? Endpoint::ShardedState : Endpoint::Ping;
    cfg.hostHeader     = fmt::format("{}:{}", cfg.host, cfg.port);
    // sessionId=0, rank=0: an unknown session parks the empty fragment in pending data.
    cfg.target = "/api/peer/shardedstateregistry/0/0";

    RaiseFdLimit();
    rlimit rl{};
    getrlimit(RLIMIT_NOFILE, &rl);

    // Build the concurrency ramp.
    std::vector<uint32_t> levels;
    if (!rampStr.empty())
    {
        std::string        token;
        std::istringstream iss(rampStr);
        while (std::getline(iss, token, ','))
        {
            if (!token.empty()) { levels.push_back(static_cast<uint32_t>(std::stoul(token))); }
        }
    }
    else if (singleNodes > 0) { levels.push_back(singleNodes); }
    else
    {
        for (uint32_t n : {32U, 64U, 128U, 256U, 512U, 1024U, 2048U, 4096U, 6144U})
        {
            if (n <= maxNodes) { levels.push_back(n); }
        }
        if (levels.empty() || levels.back() != maxNodes) { levels.push_back(maxNodes); }
    }

    fmt::print("=== ckpt_stress_test ===\n");
    fmt::print("target           : {}:{}  ({})\n", cfg.host, cfg.port, external ? "external server" : "in-process server");
    fmt::print("endpoint         : {}\n", endpointStr);
    fmt::print("msgs/node/save   : {}   (connects each node makes per save)\n", cfg.iters);
    fmt::print("saves            : {}   (ramp replayed this many times; >1 models steady-state training)\n", cfg.saves);
    fmt::print("max in-flight    : {}   <- single-host driver cap; levels above this are load- but not fully burst-modeled\n",
               cfg.maxInflight);
    fmt::print("io threads       : {}\n", ioThreads);
    fmt::print("connect timeout  : {} ms\n", cfg.connectTimeout.count());
    fmt::print("RLIMIT_NOFILE    : {} (raised to hard limit)\n", rl.rlim_cur);
    fmt::print("net.core.somaxconn        : {}   <- server listen backlog cap\n", ReadSysctl("/proc/sys/net/core/somaxconn"));
    fmt::print("tcp_max_syn_backlog       : {}\n", ReadSysctl("/proc/sys/net/ipv4/tcp_max_syn_backlog"));
    fmt::print("tcp_abort_on_overflow     : {}   (1 => refused/RST on full accept queue; 0 => drop+stall)\n",
               ReadSysctl("/proc/sys/net/ipv4/tcp_abort_on_overflow"));
    fmt::print("ramp             : {}\n\n", fmt::join(levels, ", "));

    if (!external)
    {
        CleanStaleShm();
        fmt::print("Starting in-process checkpoint web service on port {}...\n", cfg.port);
        RunInprocServer(1);
    }

    // Client io_context + worker pool.
    net::io_context          ioc;
    auto                     guard = net::make_work_guard(ioc);
    std::vector<std::thread> workers;
    workers.reserve(ioThreads);
    for (unsigned i = 0; i < ioThreads; ++i)
    {
        workers.emplace_back([&ioc] { ioc.run(); });
    }

    // Resolve once; every request reuses the endpoint list.
    tcp::resolver resolver(ioc);
    auto          eps = resolver.resolve(cfg.host, std::to_string(cfg.port));

    // Wait for the service to accept connections before measuring.
    {
        fmt::print("Waiting for the web service to become ready...\n");
        bool ready = false;
        for (int attempt = 0; attempt < 100 && !ready; ++attempt)
        {
            Stats             probe;
            std::atomic<bool> done{false};
            net::co_spawn(net::make_strand(ioc), DoOneRequest(eps, cfg, probe), [&](std::exception_ptr) { done.store(true); });
            while (!done.load()) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
            ready = probe.Get(Outcome::Ok) > 0;
            if (!ready) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
        }
        if (!ready) { fmt::print("WARNING: service did not answer a probe request; continuing anyway.\n"); }
        else
        {
            fmt::print("Service is ready.\n\n");
        }
    }

    auto printHeader = [] {
        fmt::print("{:>7} {:>10} {:>8} {:>7} {:>7} {:>7} {:>7} {:>10} {:>10} {:>10} {:>10} {:>10} {:>11}\n",
                   "nodes",
                   "attempt",
                   "ok",
                   "refused",
                   "reset",
                   "timeout",
                   "addrX",
                   "cxn_p50ms",
                   "cxn_p99ms",
                   "cxn_maxms",
                   "rtt_p99ms",
                   "wall_ms",
                   "conn/s");
        fmt::print("{:-<123}\n", "");
    };
    printHeader();

    long const overflowsBefore = ReadTcpExt("ListenOverflows");

    std::vector<LevelResult> results;
    for (uint32_t save = 1; save <= cfg.saves; ++save)
    {
        if (cfg.saves > 1)
        {
            fmt::print("── save {}/{} ──────────────────────────────────────────────────────────────\n", save, cfg.saves);
        }
        for (uint32_t nodes : levels)
        {
            auto lr = RunLevel(ioc, eps, cfg, nodes);
            results.push_back(lr);
            fmt::print("{:>7} {:>10} {:>8} {:>7} {:>7} {:>7} {:>7} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.1f} {:>11.0f}\n",
                       lr.nodes,
                       lr.attempted,
                       lr.ok,
                       lr.refused,
                       lr.reset,
                       lr.timeout,
                       lr.addrUnavail,
                       lr.connectP50,
                       lr.connectP99,
                       lr.connectMax,
                       lr.rttP99,
                       lr.wallMs,
                       lr.connPerSec);
            std::fflush(stdout);
        }
    }

    // Verdict: find the first level with any hard connection failure and the
    // throughput plateau (where conn/s stops climbing meaningfully).
    fmt::print("\n=== verdict ===\n");
    LevelResult const* firstFail      = nullptr;
    double             peakConnPerSec = 0.0;
    uint32_t           peakAt         = 0;
    long               totRefused = 0, totReset = 0, totTimeout = 0, totAddr = 0;
    for (auto const& lr : results)
    {
        long const fails = lr.refused + lr.reset + lr.timeout + lr.addrUnavail;
        if (firstFail == nullptr && fails > 0) { firstFail = &lr; }
        totRefused += lr.refused;
        totReset += lr.reset;
        totTimeout += lr.timeout;
        totAddr += lr.addrUnavail;
        if (lr.connPerSec > peakConnPerSec)
        {
            peakConnPerSec = lr.connPerSec;
            peakAt         = lr.nodes;
        }
    }
    if (firstFail != nullptr)
    {
        fmt::print("Connection failures first appear at {} concurrent nodes.\n", firstFail->nodes);
        fmt::print(
            "Totals across the run: {} refused, {} reset, {} timed out, {} addr-unavailable.\n", totRefused, totReset, totTimeout, totAddr);
        fmt::print("=> The web service does NOT scale cleanly past this point; this reproduces the large-scale symptom.\n");
        // Attribute the dominant failure mode so the user knows where to look.
        if (totTimeout > 0)
        {
            fmt::print("   - timeouts: connects stalled behind node:0's serialized accept loop / full backlog (server-side).\n");
        }
        if (totRefused > 0)
        {
            fmt::print("   - refused: node:0 RST'd connects (accept queue full with abort-on-overflow, or not listening) (server-side).\n");
        }
        if (totAddr > 0)
        {
            fmt::print("   - addr-unavailable: client ran out of ephemeral ports — the connect-per-message design burns a port\n"
                       "     (then TIME_WAIT) per notify/data message; at scale this bites the *sender* too (client-side).\n");
        }
    }
    else
    {
        fmt::print("No hard connection failures up to {} concurrent nodes on this host.\n", levels.back());
        fmt::print("Watch cxn_p99ms / rtt_p99ms and conn/s: a rising tail with flat throughput is the serialized-accept plateau.\n");
    }
    fmt::print("Peak throughput: {:.0f} conn/s at {} nodes (serial accept+handle ceiling).\n", peakConnPerSec, peakAt);
    long const overflowsAfter = ReadTcpExt("ListenOverflows");
    if (overflowsBefore >= 0 && overflowsAfter >= 0)
    {
        long const delta = overflowsAfter - overflowsBefore;
        fmt::print("Kernel accept-queue overflows during run: {}  (TcpExt:ListenOverflows delta)\n", delta);
        if (delta > 0)
        {
            fmt::print("=> node:0's listen backlog ({}) overflowed: the serial accept loop could not drain connections\n"
                       "   fast enough, so the kernel dropped/stalled incoming connects. This is the mechanism behind\n"
                       "   the connection-refused/timeout failures reported at large scale.\n",
                       ReadSysctl("/proc/sys/net/core/somaxconn"));
        }
    }
    std::fflush(stdout);
    int const rc = (failOnErrors && firstFail != nullptr) ? 2 : 0;

    guard.reset();
    ioc.stop();
    for (auto& w : workers) { w.join(); }

    if (!external)
    {
        // The in-process server's teardown can wedge on joining its shm queue
        // reader threads. The measurement is already reported, so attempt a
        // graceful stop (which unlinks the shm) but never let it hang the tool:
        // if it does not finish promptly, drop the stale shm and force-exit.
        std::atomic<bool> stopped{false};
        std::thread       stopper([&] {
            StopInprocServer();
            stopped.store(true);
        });
        for (int i = 0; i < 50 && !stopped.load(); ++i) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
        if (stopped.load()) { stopper.join(); }
        else
        {
            stopper.detach();
            CleanStaleShm();
            std::fflush(stdout);
            std::_Exit(rc);
        }
    }
    return rc;
} catch (std::exception const& e)
{
    fmt::print(stderr, "ckpt_stress_test error: {}\n", e.what());
    return 1;
}
