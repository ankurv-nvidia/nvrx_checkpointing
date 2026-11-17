#include "svcref.h"
#include "CommonMacros.h"

SUPPRESS_WARNINGS_START
SUPPRESS_CLANG_WARNING("-Wmissing-noreturn")
SUPPRESS_CLANG_WARNING("-Wunused-parameter")
SvcRef& SvcRef::Instance_()
{
    // Its ok all local servers can share the same svcref
    // They all spin down together
    static auto* Instance = new SvcRef();    // NOLINT(cppcoreguidelines-owning-memory)
    return *Instance;
}

std::shared_ptr<SvcRef::InProc> SvcRef::AcquireInProc_()
{ TODO("not implemented"); }

std::shared_ptr<SvcRef::LocalProcCli> SvcRef::CliAcquireLocalProc_(uint16_t serverIndex)
{ TODO("not implemented"); }

std::shared_ptr<SvcRef::RemoteProcCli> SvcRef::CliAcquireRemoteProc_(std::string peer)
{ TODO("not implemented"); }

uint64_t SvcRef::SvrAcquireLocalProcRef_(uint64_t pid)
{ TODO("not implemented"); }

void SvcRef::SvrReleaseLocalProcRef_(uint64_t pid, uint64_t token)
{ TODO("not implemented"); }

uint64_t SvcRef::SvrAcquireRemoteProcRef_(std::string peer)
{ TODO("not implemented"); }

void SvcRef::SvrReleaseRemoteProcRef_(std::string_view const& peer, uint64_t token)
{ TODO("not implemented"); }

SUPPRESS_WARNINGS_END
