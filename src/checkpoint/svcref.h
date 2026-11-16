#pragma once
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

struct SvcRef
{
    public:
    struct InProc
    {};
    struct LocalProcCli
    {
        uint64_t pid{};
        uint64_t token{};
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
    };

    private:
    auto AcquireInProc_()
    {
        auto inProc = std::make_shared<InProc>();
        {
            std::scoped_lock lock(_mutex);
            _inProc.push_back(inProc);
        }
        return inProc;
    }

    auto CliAcquireLocalProc_(int64_t pid)
    {
        auto localProc = std::make_shared<LocalProcCli>();
        localProc->pid = pid;
        {
            std::scoped_lock lock(_mutex);
            _localProc.push_back(localProc);
        }
        return localProc;
    }

    auto CliAcquireRemoteProc_()
    {
        auto remoteProc = std::make_shared<RemoteProcCli>();
        {
            std::scoped_lock lock(_mutex);
            _remoteProc.push_back(remoteProc);
        }
        return remoteProc;
    }

    uint64_t SvrAcquireLocalProcRef_(uint64_t pid);
    void     SvrReleaseLocalProcRef_(uint64_t pid, uint64_t token);
    uint64_t SvrAcquireRemoteProcRef_(std::string peer);
    void     SvrReleaseRemoteProcRef_(std::string_view const& peer, uint64_t token);

    public:
    static auto AcquireInProc() { return Instance_().AcquireInProc_(); }
    static auto CliAcquireLocalProc(int64_t pid) { return Instance_().CliAcquireLocalProc_(pid); }
    static auto CliAcquireRemoteProc() { return Instance_().CliAcquireRemoteProc_(); }
    static auto SvrAcquireLocalProcRef(uint64_t pid) { return Instance_().SvrAcquireLocalProcRef_(pid); }
    static void SvrReleaseLocalProcRef(uint64_t pid, uint64_t token) { Instance_().SvrReleaseLocalProcRef_(pid, token); }
    static auto SvrAcquireRemoteProcRef(std::string peer) { return Instance_().SvrAcquireRemoteProcRef_(std::move(peer)); }
    static auto SvrReleaseRemoteProcRef(std::string_view const& peer, uint64_t token) { Instance_().SvrReleaseRemoteProcRef_(peer, token); }

    private:
    static SvcRef& Instance_();

    uint32_t                                _token;
    std::mutex                              _mutex;
    std::list<std::weak_ptr<InProc>>        _inProc;
    std::list<std::weak_ptr<LocalProcCli>>  _localProc;
    std::list<std::weak_ptr<RemoteProcCli>> _remoteProc;
};