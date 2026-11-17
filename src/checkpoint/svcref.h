#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

struct SvcRef
{
    public:
    using Clock = std::chrono::steady_clock;

    static constexpr Clock::duration  DefaultStartupGrace = std::chrono::seconds{5};
    static constexpr std::string_view PeerHeader          = "X-Native-Ckpt-Peer";

    struct InProc
    {};
    struct LocalProcCli
    {
        uint64_t pid{};
        uint64_t token{};
        uint64_t generation{};
    };
    struct RemoteProcCli
    {
        uint64_t    token{};
        std::string peer;
    };

    private:
    struct LocalProcSrv
    {
        uint64_t pid{};
        uint64_t token{};
    };

    struct RemoteProcSrv
    {
        uint64_t    token{};
        std::string peer;
        bool        answering{true};
    };

    private:
    std::shared_ptr<InProc>        AcquireInProc_();
    std::shared_ptr<LocalProcCli>  CliAcquireLocalProc_(uint16_t serverIndex);
    std::shared_ptr<RemoteProcCli> CliAcquireRemoteProc_(std::string peer, std::string const& address, Clock::duration timeout);

    uint64_t SvrAcquireLocalProcRef_(uint64_t pid);
    void     SvrReleaseLocalProcRef_(uint64_t pid, uint64_t token);
    uint64_t SvrAcquireRemoteProcRef_(std::string peer);
    void     SvrReleaseRemoteProcRef_(std::string_view const& peer, uint64_t token);

    public:
    static auto AcquireInProc() { return Instance_().AcquireInProc_(); }
    static auto CliAcquireLocalProc(uint16_t serverIndex) { return Instance_().CliAcquireLocalProc_(serverIndex); }
    static auto CliAcquireRemoteProc(std::string peer, std::string const& address, Clock::duration timeout)
    { return Instance_().CliAcquireRemoteProc_(std::move(peer), address, timeout); }

    static auto SvrAcquireLocalProcRef(uint64_t pid) { return Instance_().SvrAcquireLocalProcRef_(pid); }
    static void SvrReleaseLocalProcRef(uint64_t pid, uint64_t token) { Instance_().SvrReleaseLocalProcRef_(pid, token); }
    static auto SvrAcquireRemoteProcRef(std::string peer) { return Instance_().SvrAcquireRemoteProcRef_(std::move(peer)); }
    static auto SvrReleaseRemoteProcRef(std::string_view const& peer, uint64_t token) { Instance_().SvrReleaseRemoteProcRef_(peer, token); }

    // Returns once a service answers for the server index, hosting or launching one when none does.
    static void CliEnsureLocalProc(uint16_t serverIndex);

    // Drops every reference a pid holds, such as an exiting host's own.
    static void SvrReleaseLocalProcRefs(uint64_t pid);

    // Starts serving references for a daemon of this process; a daemon already serving keeps the current ones.
    static void BeginService(Clock::duration startupGrace = DefaultStartupGrace, Clock::duration scanInterval = ConfiguredScanInterval());

    // Returns once no reference counts or a stop is requested; the service is closing from then on.
    static void WaitUntilIdle(std::atomic<bool> const& stopRequested);

    [[nodiscard]] static bool            Closing();
    [[nodiscard]] static bool            HasInProc();
    [[nodiscard]] static uint64_t        Generation();
    [[nodiscard]] static uint64_t        CurrentPid();
    [[nodiscard]] static bool            ProcessAlive(uint64_t pid);
    [[nodiscard]] static bool            Answers(uint16_t port);
    [[nodiscard]] static Clock::duration ConfiguredScanInterval();
    static void                          Wake();

    // The returned call holds an in-process reference from now until it returns.
    template <typename TCall> static auto Bind(TCall&& call)
    {
        return [reference = AcquireInProc(), work = std::forward<TCall>(call)]() mutable {
            auto const held = std::move(reference);
            work();
        };
    }

    template <typename TCall> static std::future<void> Async(TCall&& call)
    { return std::async(std::launch::async, Bind(std::forward<TCall>(call))); }

    private:
    static SvcRef& Instance_();

    bool Live_();
    void Changed_();

    // A daemon can acquire both cli and srv references
    // Both must be empty to safely exit or release resources.
    std::mutex                              _mutex;
    std::list<std::weak_ptr<InProc>>        _inProc;
    std::list<std::weak_ptr<LocalProcCli>>  _localProcCli;
    std::list<std::weak_ptr<RemoteProcCli>> _remoteProcCli;
    std::list<LocalProcSrv>                 _localProcSrv;
    std::list<RemoteProcSrv>                _remoteProcSrv;

    std::condition_variable _changed;
    Clock::duration         _startupGrace{DefaultStartupGrace};
    Clock::duration         _scanInterval{std::chrono::seconds{5}};
    uint64_t                _token{};
    uint64_t                _generation{};
    uint64_t                _revision{};
    bool                    _serving{};
    bool                    _linkUpstream{};
    bool                    _owned{};
    bool                    _closing{};
    bool                    _woken{};
};
