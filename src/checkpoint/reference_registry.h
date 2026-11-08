#pragma once

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <sys/types.h>

#ifdef __APPLE__
#include <libproc.h>
#include <sys/proc.h>
#endif

namespace checkpoint
{
struct ProcessIdentity
{
    pid_t    pid{};
    uint64_t started{};

    bool operator==(ProcessIdentity const&) const = default;

    static std::optional<ProcessIdentity> Read(pid_t pid)
    {
        if (pid <= 0) { return std::nullopt; }
#ifdef __linux__
        std::ifstream file("/proc/" + std::to_string(pid) + "/stat");
        std::string   line;
        if (!std::getline(file, line)) { return std::nullopt; }
        auto const close = line.rfind(')');
        if (close == std::string::npos) { return std::nullopt; }
        std::istringstream fields(line.substr(close + 1));
        char               state{};
        fields >> state;
        if (state == 'Z' || state == 'X' || state == 'x') { return std::nullopt; }
        std::string ignored;
        for (int field = 4; field < 22; ++field) { fields >> ignored; }
        ProcessIdentity identity{.pid = pid};
        if (!(fields >> identity.started)) { return std::nullopt; }
        return identity;
#elif defined(__APPLE__)
        proc_bsdinfo info{};
        if (::proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, static_cast<int>(sizeof(info))) != static_cast<int>(sizeof(info))
            || info.pbi_status == SZOMB)
        {
            return std::nullopt;
        }
        static constexpr uint64_t MicrosPerSecond = 1'000'000;
        return ProcessIdentity{.pid = pid, .started = info.pbi_start_tvsec * MicrosPerSecond + info.pbi_start_tvusec};
#else
        if (::kill(pid, 0) != 0 && errno != EPERM) { return std::nullopt; }
        return ProcessIdentity{.pid = pid};
#endif
    }

    bool IsAlive() const { return Read(pid) == *this; }
};

class ReferenceRegistry
{
    public:
    using SessionId                      = uint32_t;
    using Lease                          = std::shared_ptr<void>;
    using Clock                          = std::chrono::steady_clock;
    static constexpr SessionId NoSession = UINT32_MAX;

    private:
    struct References
    {
        std::weak_ptr<void>                    self;
        std::map<std::string, ProcessIdentity> local;
        std::map<std::string, std::string>     remote;
        std::set<std::string>                  releasedLocal;
        std::set<std::string>                  releasedRemote;
    };

    mutable std::mutex              _mutex;
    std::map<SessionId, References> _sessions;
    std::set<SessionId>             _closedSessions;
    bool                            _accepting{true};
    bool                            _acquired{};
    Clock::time_point               _started{Clock::now()};

    public:
    Lease AcquireSelf(SessionId session = NoSession, bool control = false)
    {
        std::scoped_lock lock(_mutex);
        if (!_accepting && !control) { return {}; }
        if (session != NoSession && _closedSessions.contains(session)) { return {}; }
        auto& self  = _sessions[session].self;
        auto  lease = self.lock();
        if (!lease)
        {
            lease = std::make_shared<char>();
            self  = lease;
        }
        return lease;
    }

    void AcquireLocal(SessionId session, std::string const& token, pid_t pid)
    {
        auto const identity = ProcessIdentity::Read(pid);
        if (!identity) { throw std::runtime_error("Checkpoint client process is not alive"); }
        std::scoped_lock lock(_mutex);
        if (!_accepting) { throw std::runtime_error("Checkpoint daemon is stopping"); }
        auto& refs = _sessions[session];
        if (refs.releasedLocal.contains(token)) { throw std::runtime_error("Checkpoint reference is already released"); }
        auto const [it, inserted] = refs.local.emplace(token, *identity);
        if (!inserted && it->second != *identity) { throw std::runtime_error("Checkpoint reference token has a different owner"); }
        _closedSessions.erase(session);
        _acquired = true;
    }

    void ReleaseLocal(SessionId session, std::string const& token)
    {
        std::scoped_lock lock(_mutex);
        auto&            refs = _sessions[session];
        refs.releasedLocal.insert(token);
        refs.local.erase(token);
    }

    void AcquireRemote(SessionId session, std::string const& token, std::string const& peer)
    {
        std::scoped_lock lock(_mutex);
        if (!_accepting) { throw std::runtime_error("Checkpoint daemon is stopping"); }
        auto& refs = _sessions[session];
        if (refs.releasedRemote.contains(token)) { throw std::runtime_error("Checkpoint reference is already released"); }
        auto const [it, inserted] = refs.remote.emplace(token, peer);
        if (!inserted && it->second != peer) { throw std::runtime_error("Checkpoint reference token has a different owner"); }
        _closedSessions.erase(session);
        _acquired = true;
    }

    void ReleaseRemote(SessionId session, std::string const& token)
    {
        std::scoped_lock lock(_mutex);
        auto&            refs = _sessions[session];
        refs.releasedRemote.insert(token);
        refs.remote.erase(token);
    }

    bool HasUnassignedWork() const
    {
        std::scoped_lock lock(_mutex);
        auto const       it = _sessions.find(NoSession);
        return it != _sessions.end() && !it->second.self.expired();
    }

    bool IsAccepting() const
    {
        std::scoped_lock lock(_mutex);
        return _accepting;
    }

    bool TryCloseSession(SessionId session)
    {
        std::scoped_lock lock(_mutex);
        auto const       unassigned = _sessions.find(NoSession);
        if (unassigned != _sessions.end() && !unassigned->second.self.expired()) { return false; }
        auto const it = _sessions.find(session);
        if (it != _sessions.end())
        {
            auto const& refs = it->second;
            if (!refs.self.expired() || !refs.local.empty() || !refs.remote.empty()) { return false; }
        }
        _closedSessions.insert(session);
        return true;
    }

    template <typename TPing> bool HasValidReferences(TPing const& ping, std::optional<SessionId> session = std::nullopt)
    {
        while (true)
        {
            std::vector<std::tuple<SessionId, std::string, std::string>> remotes;
            {
                std::scoped_lock lock(_mutex);
                for (auto const& [id, refs] : _sessions)
                {
                    if ((!session || id == *session) && !refs.self.expired()) { return true; }
                }
                for (auto& [id, refs] : _sessions)
                {
                    if (session && id != *session) { continue; }
                    for (auto it = refs.local.begin(); it != refs.local.end();)
                    {
                        if (it->second.IsAlive()) { return true; }
                        refs.releasedLocal.insert(it->first);
                        it = refs.local.erase(it);
                    }
                }
                for (auto const& [id, refs] : _sessions)
                {
                    if (session && id != *session) { continue; }
                    for (auto const& [token, peer] : refs.remote) { remotes.emplace_back(id, token, peer); }
                }
                if (remotes.empty()) { return false; }
            }
            for (auto const& [id, token, peer] : remotes)
            {
                bool const       alive = ping(peer);
                std::scoped_lock lock(_mutex);
                auto&            refs = _sessions.at(id).remote;
                auto             it   = refs.find(token);
                if (it == refs.end() || it->second != peer) { continue; }
                if (alive) { return true; }
                _sessions.at(id).releasedRemote.insert(token);
                refs.erase(it);
            }
        }
    }

    bool TryRetire(Clock::time_point now = Clock::now())
    {
        static constexpr auto StartupGrace = std::chrono::seconds{5};
        std::scoped_lock      lock(_mutex);
        if (!_acquired && now - _started < StartupGrace) { return false; }
        for (auto const& [id, refs] : _sessions)
        {
            static_cast<void>(id);
            if (!refs.self.expired() || !refs.local.empty() || !refs.remote.empty()) { return false; }
        }
        _accepting = false;
        return true;
    }
};
}    // namespace checkpoint
