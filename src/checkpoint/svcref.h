#pragma once
#include "CommonMacros.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

struct SvcRef
{
    public:
    static constexpr uint32_t AllSessions = UINT32_MAX;

    struct InProc
    {
        InProc(SvcRef& owner LFTBND, std::string_view name, uint32_t session, bool control);
        ~InProc();
        CLASS_DELETE_COPY_AND_MOVE(InProc);

        std::string const context;

        private:
        friend struct SvcRef;
        SvcRef*        _owner{};
        uint64_t const _pid;
        uint32_t const _session;
        bool const     _tracked;
    };
    struct LocalProcCli
    {
        uint64_t pid{};
        uint64_t token{};
        uint64_t generation{};
        ~LocalProcCli();
        void Close() noexcept;

        private:
        friend struct SvcRef;
        uint16_t   _index{};
        std::mutex _mutex;
    };
    struct RemoteProcCli
    {
        uint64_t    token{};
        std::string peer;
        ~RemoteProcCli();
        void Close() noexcept;

        private:
        friend struct SvcRef;
        uint64_t    _pid{};
        uint16_t    _index{};
        uint32_t    _session{AllSessions};
        std::string _address;
        bool        _managed{};
        std::mutex  _mutex;
    };

    private:
    struct LocalProcSrv
    {
        uint64_t pid{};
        uint64_t token{};
        uint32_t session{AllSessions};
    };
    struct RemoteProcSrv
    {
        uint64_t    token{};
        std::string peer;
        uint64_t    generation{};
        uint32_t    session{AllSessions};
    };
    struct Link
    {
        std::mutex                     mutex;
        std::shared_ptr<RemoteProcCli> reference;
    };
    struct Instances;
    using Clock = std::chrono::steady_clock;

    explicit SvcRef(uint16_t index);
    void                           EnsureService_();
    std::shared_ptr<InProc>        AcquireInProc_(std::string_view context, uint32_t session, bool control);
    std::shared_ptr<LocalProcCli>  CliAcquireLocalProc_(uint32_t session, std::string coordinator, std::string address);
    std::shared_ptr<RemoteProcCli> CliAcquireRemoteProc_(std::string peer, std::string address, uint32_t session, bool managed);
    uint64_t SvrAcquireLocalProcRef_(uint64_t pid, uint32_t session, std::string coordinator, std::string address, uint64_t token);
    void     SvrReleaseLocalProcRef_(uint64_t pid, uint64_t token);
    uint64_t SvrAcquireRemoteProcRef_(std::string peer, uint64_t generation, uint32_t session, uint64_t token);
    void     SvrReleaseRemoteProcRef_(std::string_view const& peer, uint64_t token);
    bool     HasReferences_(uint32_t session);
    void     ReleaseRemoteClients_(bool all, std::function<bool()> const& drained = {});
    void     Release_(std::string peer, std::string target, std::string body = {}) noexcept;
    void     RetryReleases_();
    void     CheckOpen_();
    uint64_t Token_();

    public:
    static auto AcquireInProc(std::string_view context = {}, uint32_t session = AllSessions, uint16_t index = 0, bool control = false)
    { return Instance_(index).AcquireInProc_(context, session, control); }
    static auto
    CliAcquireLocalProc(uint16_t index = 0, uint32_t session = AllSessions, std::string coordinator = {}, std::string address = {})
    { return Instance_(index).CliAcquireLocalProc_(session, std::move(coordinator), std::move(address)); }
    static auto CliAcquireRemoteProc(std::string peer = {}, std::string address = {}, uint32_t session = AllSessions, uint16_t index = 0)
    { return Instance_(index).CliAcquireRemoteProc_(std::move(peer), std::move(address), session, false); }

    static auto SvrAcquireLocalProcRef(uint64_t    pid,
                                       uint16_t    index       = 0,
                                       uint32_t    session     = AllSessions,
                                       std::string coordinator = {},
                                       std::string address     = {},
                                       uint64_t    token       = 0)
    { return Instance_(index).SvrAcquireLocalProcRef_(pid, session, std::move(coordinator), std::move(address), token); }
    static void SvrReleaseLocalProcRef(uint64_t pid, uint64_t token, uint16_t index = 0)
    { Instance_(index).SvrReleaseLocalProcRef_(pid, token); }
    static auto SvrAcquireRemoteProcRef(std::string peer,
                                        uint64_t    generation = 0,
                                        uint16_t    index      = 0,
                                        uint32_t    session    = AllSessions,
                                        uint64_t    token      = 0)
    { return Instance_(index).SvrAcquireRemoteProcRef_(std::move(peer), generation, session, token); }
    static void SvrReleaseRemoteProcRef(std::string_view const& peer, uint64_t token, uint16_t index = 0)
    { Instance_(index).SvrReleaseRemoteProcRef_(peer, token); }

    template <typename TCall>
    static auto Bind(std::string_view context, TCall&& call, uint32_t session = AllSessions, uint16_t index = 0, bool control = false)
    {
        return [reference = AcquireInProc(context, session, index, control), work = std::forward<TCall>(call)] mutable {
            auto held = std::move(reference);
            if (held) { work(); }
        };
    }
    static void     EnsureService(uint16_t index = 0) { Instance_(index).EnsureService_(); }
    static void     SvrStart(uint16_t index = 0);
    static uint64_t SvrGeneration(uint16_t index = 0);
    static bool     SvrClosing(uint16_t index = 0);
    static void     WaitForShutdown(uint16_t index, std::atomic<bool> const& stopRequested, std::function<bool()> const& drained = {});
    static void     DrainRequests(uint16_t index = 0);
    static void     Seal(uint16_t index = 0);
    static void     ReleaseRemoteClients(uint16_t index = 0) { Instance_(index).ReleaseRemoteClients_(true); }
    static void     CloseLocalProcClients();

    private:
    static SvcRef&    Instance_(uint16_t index = 0);
    static Instances& Instances_();

    uint32_t                                                       _token{};
    uint16_t                                                       _index{};
    uint64_t                                                       _generation{};
    std::mutex                                                     _mutex;
    std::mutex                                                     _startMutex;
    std::condition_variable                                        _changed;
    std::map<uint32_t, uint64_t>                                   _inProc;
    std::list<std::weak_ptr<LocalProcCli>>                         _localProcCli;
    std::list<std::weak_ptr<RemoteProcCli>>                        _remoteProcCli;
    std::list<std::shared_ptr<LocalProcSrv>>                       _localProcSrv;
    std::list<std::shared_ptr<RemoteProcSrv>>                      _remoteProcSrv;
    std::map<uint32_t, std::shared_ptr<Link>>                      _links;
    std::vector<std::tuple<std::string, std::string, std::string>> _pending;
    std::unordered_set<uint64_t>                                   _released;
    std::unordered_set<uint64_t>                                   _exitGenerations;
    uint64_t                                                       _revision{};
    uint64_t                                                       _controlRevision{};
    bool                                                           _owned{};
    bool                                                           _closing{};
    bool                                                           _sealed{};
    Clock::time_point                                              _started{Clock::now()};
};
