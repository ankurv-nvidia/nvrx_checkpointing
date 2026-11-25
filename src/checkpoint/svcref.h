#pragma once
#include "CommonMacros.h"

#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

struct SvcRef
{
    public:
    struct InProc
    {
        InProc();
        ~InProc();
        CLASS_DEFAULT_COPY_AND_MOVE(InProc);
    };

    struct LocalProcCli
    {
        LocalProcCli();
        ~LocalProcCli();
        CLASS_DEFAULT_COPY_AND_MOVE(LocalProcCli);
        uint64_t pid{};
        uint64_t token{};
    };
    struct RemoteProcCli
    {
        RemoteProcCli();
        ~RemoteProcCli();
        CLASS_DEFAULT_COPY_AND_MOVE(RemoteProcCli);
        uint64_t    token{};
        std::string peer;
    };

    private:
    struct LocalProcSrv
    {
        LocalProcSrv();
        ~LocalProcSrv();
        CLASS_DEFAULT_COPY_AND_MOVE(LocalProcSrv);
        uint64_t pid{};
        uint64_t token{};
    };

    struct RemoteProcSrv
    {
        RemoteProcSrv();
        ~RemoteProcSrv();
        CLASS_DEFAULT_COPY_AND_MOVE(RemoteProcSrv);
        uint64_t    token{};
        std::string peer;
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

    private:
    static SvcRef& Instance_();

    // A daemon can acquire both cli and srv references
    // Both must be empty to safely exit or release resources.
    std::mutex                                _mutex;
    std::list<std::weak_ptr<LocalProcCli>>    _localProcCli;
    std::list<std::weak_ptr<RemoteProcCli>>   _remoteProcCli;
    std::list<std::shared_ptr<LocalProcSrv>>  _localProcSrv;
    std::list<std::shared_ptr<RemoteProcSrv>> _remoteProcSrv;
};
