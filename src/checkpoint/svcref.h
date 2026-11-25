#pragma once
#include "CommonMacros.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
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
    {
        InProc();
        ~InProc();
        CLASS_DELETE_COPY_AND_MOVE(InProc);
    };

    struct LocalProcCli
    {
        explicit LocalProcCli(uint16_t serverIndex);
        ~LocalProcCli();
        CLASS_DELETE_COPY_AND_MOVE(LocalProcCli);
        uint64_t pid{};
        uint64_t token{};
        uint64_t generation{};
        uint16_t port{};
    };
    struct RemoteProcCli
    {
        RemoteProcCli(std::string peer);
        ~RemoteProcCli();
        CLASS_DELETE_COPY_AND_MOVE(RemoteProcCli);
        uint64_t    token{};
        std::string peer;
        std::string self;
        uint64_t    pid{};
    };

    private:
    struct LocalProcSrv
    {
        LocalProcSrv();
        ~LocalProcSrv();
        CLASS_DELETE_COPY_AND_MOVE(LocalProcSrv);
        uint64_t pid{};
        uint64_t token{};
    };

    struct RemoteProcSrv
    {
        RemoteProcSrv();
        ~RemoteProcSrv();
        CLASS_DELETE_COPY_AND_MOVE(RemoteProcSrv);
        uint64_t    token{};
        std::string peer;
        bool        answering{true};
    };

    private:
    std::shared_ptr<InProc>        AcquireInProc_();
    std::shared_ptr<LocalProcCli>  CliAcquireLocalProc_(uint16_t serverIndex);
    std::shared_ptr<RemoteProcCli> CliAcquireRemoteProc_(std::string peer);

    uint64_t SvrAcquireLocalProcRef_(uint64_t pid);
    void     SvrReleaseLocalProcRef_(uint64_t pid, uint64_t token);
    uint64_t SvrAcquireRemoteProcRef_(std::string peer);
    void     SvrReleaseRemoteProcRef_(std::string_view const& peer, uint64_t token);

    public:
    static auto AcquireInProc() { return Instance_().AcquireInProc_(); }
    static auto CliAcquireLocalProc(uint16_t serverIndex) { return Instance_().CliAcquireLocalProc_(serverIndex); }
    static auto CliAcquireRemoteProc(std::string peer) { return Instance_().CliAcquireRemoteProc_(std::move(peer)); }

    static auto SvrAcquireLocalProcRef(uint64_t pid) { return Instance_().SvrAcquireLocalProcRef_(pid); }
    static void SvrReleaseLocalProcRef(uint64_t pid, uint64_t token) { Instance_().SvrReleaseLocalProcRef_(pid, token); }
    static auto SvrAcquireRemoteProcRef(std::string peer) { return Instance_().SvrAcquireRemoteProcRef_(std::move(peer)); }
    static auto SvrReleaseRemoteProcRef(std::string_view const& peer, uint64_t token) { Instance_().SvrReleaseRemoteProcRef_(peer, token); }

    // Drops every reference a pid holds, such as an exiting host's own.
    static void SvrReleaseLocalProcRefs(uint64_t pid);

    // Records the address peers reach this process's daemon at; the first one recorded for a service stays.
    static void RecordSelfAddress(std::string address);

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

    private:
    static SvcRef& Instance_();

    bool Live_();
    void Changed_();

    // A daemon can acquire both cli and srv references
    // Both must be empty to safely exit or release resources.
    std::mutex                                _mutex;
    std::list<std::shared_ptr<LocalProcSrv>>  _localProcSrv;
    std::list<std::shared_ptr<RemoteProcSrv>> _remoteProcSrv;

    std::condition_variable _changed;
    std::string             _selfAddress;
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
