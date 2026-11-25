#include "svcref.h"
#include "net_utils.h"
#include "service_client.h"

#include <algorithm>
#include <boost/dll/runtime_symbol_info.hpp>
#include <boost/process.hpp>
#include <boost/process/execute.hpp>

#include <atomic>

static void SvcPing(uint16_t index)
{
    SUPPRESS_WARNINGS_START
    SUPPRESS_CLANG_WARNING("-Wexit-time-destructors")
    static std::mutex PingMutex;
    SUPPRESS_WARNINGS_END

    std::scoped_lock lock(PingMutex);
    auto             selfLoc   = boost::dll::this_line_location();
    auto             svcExeLoc = selfLoc.parent_path() / "checkpoint_server";
    if (TryConnectToCheckpointService(LocalSvrPeer, index)) { return; }
    if (checkpoint::config::GetConfigValue<checkpoint::config::SvcDisableAutoLaunch>())
    {
        throw std::runtime_error("Externally managed checkpoint service is unavailable");
    }

    try
    {
        boost::asio::io_context    ctx;
        boost::asio::readable_pipe output(ctx);
        boost::process::process    proc(ctx, svcExeLoc, {}, boost::process::process_stdio{.in = {}, .out = output, .err = stderr});
        std::string                line;
        boost::system::error_code  ec;
        do { boost::asio::read_until(output, boost::asio::dynamic_buffer(line), '\n', ec); } while (ec == boost::asio::error::interrupted);
        proc.detach();
    } catch (std::exception const& ex)
    {
        fmt::print(stderr,
                   "[native_ckpt:cli][ERROR] Exception in SvcPing from {} svcbin={}: {}\n",
                   selfLoc.string(),
                   svcExeLoc.string(),
                   ex.what());
        throw;
    }

    // Both the elected daemon and losing candidates write READY only after the
    // elected service answers a ping. The two short attempts inside TryConnect
    // cover a final scheduling edge without a second readiness policy here.
    if (!TryConnectToCheckpointService(LocalSvrPeer, index))
    {
        throw std::runtime_error("Failed to start or connect to checkpoint service");
    }
}

static void WaitForRemotePeerPing(std::string_view const& peer)
{
    if (!TryConnectToCheckpointService(peer, 0)) { throw std::runtime_error("Failed to connect to remote peer"); }
}

static std::atomic<uint64_t> InProcCounter{0};           // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
static std::atomic<uint64_t> LocalProcCounterCli{0};     // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
static std::atomic<uint64_t> RemoteProcCounterCli{0};    // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
static std::atomic<uint64_t> LocalProcCounterSrv{0};     // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
static std::atomic<uint64_t> RemoteProcCounterSrv{0};    // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

SvcRef::InProc::InProc()
{ InProcCounter++; }
SvcRef::InProc::~InProc()
{ InProcCounter--; }

SvcRef::LocalProcSrv::LocalProcSrv()
{ LocalProcCounterSrv++; }
SvcRef::LocalProcSrv::~LocalProcSrv()
{ LocalProcCounterSrv--; }
SvcRef::RemoteProcSrv::RemoteProcSrv()
{ RemoteProcCounterSrv++; }
SvcRef::RemoteProcSrv::~RemoteProcSrv()
{ RemoteProcCounterSrv--; }

SvcRef::LocalProcCli::LocalProcCli(uint16_t serverIndex)
{
    if (LocalProcCounterCli == 0) { SvcPing(serverIndex); }
    LocalProcCounterCli++;
    // TODO_NOMERGE: Send a increment-ref-count msg to LocalSrvPeer and store the returned tag
}

SvcRef::LocalProcCli::~LocalProcCli()
{
    LocalProcCounterCli--;
    // TODO_NOMERGE: Send a decrement-ref-count msg to LocalSrvPeer along with the stored tag
}

SvcRef::RemoteProcCli::RemoteProcCli(std::string peerIn) : peer(std::move(peerIn))
{
    if (RemoteProcCounterCli == 0) { WaitForRemotePeerPing(peer); }    // Assuming 0 as the server index for simplicity
    RemoteProcCounterCli++;
    // TODO_NOMERGE: Send a increment-ref-count msg to RemoteSrvPeer and store the returned tag
}

SvcRef::RemoteProcCli::~RemoteProcCli()
{
    RemoteProcCounterCli--;
    // TODO_NOMERGE: Send a decrement-ref-count msg to RemoteSrvPeer along with the stored tag
}

std::shared_ptr<SvcRef::InProc> SvcRef::AcquireInProc_()
{
    std::scoped_lock lock(_mutex);
    return std::make_shared<SvcRef::InProc>();
}

std::shared_ptr<SvcRef::LocalProcCli> SvcRef::CliAcquireLocalProc_(uint16_t serverIndex)
{
    std::scoped_lock lock(_mutex);
    return std::make_shared<SvcRef::LocalProcCli>(serverIndex);
}

std::shared_ptr<SvcRef::RemoteProcCli> SvcRef::CliAcquireRemoteProc_(std::string peer)
{
    std::scoped_lock lock(_mutex);
    return std::make_shared<SvcRef::RemoteProcCli>(std::move(peer));
}

uint64_t SvcRef::SvrAcquireLocalProcRef_(uint64_t pid)
{
    std::scoped_lock lock(_mutex);
    auto             ref = std::make_shared<SvcRef::LocalProcSrv>();
    _localProcSrv.push_back(ref);
    ref->pid   = pid;
    ref->token = LocalProcCounterSrv.load();
    return ref->token;
}

void SvcRef::SvrReleaseLocalProcRef_(uint64_t pid, uint64_t token)
{
    std::scoped_lock lock(_mutex);
    auto             ref = std::ranges::find_if(_localProcSrv, [&](auto const& r) { return r->token == token; });
    if (ref == _localProcSrv.end()) throw std::runtime_error("LocalProcSrv reference not found");
    if ((*ref)->pid != pid) throw std::runtime_error("LocalProcSrv pid mismatch");
    _localProcSrv.erase(ref);
}

uint64_t SvcRef::SvrAcquireRemoteProcRef_(std::string peer)
{
    std::scoped_lock lock(_mutex);
    auto             ref = std::make_shared<SvcRef::RemoteProcSrv>();
    ref->peer            = std::move(peer);
    ref->token           = RemoteProcCounterSrv.load();
    _remoteProcSrv.push_back(ref);
    return ref->token;
}

void SvcRef::SvrReleaseRemoteProcRef_(std::string_view const& peer, uint64_t token)
{
    std::scoped_lock lock(_mutex);
    auto             ref = std::ranges::find_if(_remoteProcSrv, [&](auto const& r) { return r->token == token; });
    if (ref == _remoteProcSrv.end()) throw std::runtime_error("RemoteProcSrv reference not found");
    if ((*ref)->peer != peer) throw std::runtime_error("RemoteProcSrv peer mismatch");
    _remoteProcSrv.erase(ref);
}

SvcRef& SvcRef::Instance_()
{
    SUPPRESS_WARNINGS_START
    SUPPRESS_CLANG_WARNING("-Wexit-time-destructors")
    static SvcRef Instance;
    SUPPRESS_WARNINGS_END
    return Instance;
}
