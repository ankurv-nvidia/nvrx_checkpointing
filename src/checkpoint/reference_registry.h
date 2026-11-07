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
        size_t                                 self{};
        std::map<std::string, ProcessIdentity> local;
        std::map<std::string, std::string>     remote;
        std::set<std::string>                  releasedLocal;
        std::set<std::string>                  releasedRemote;
    };

    struct State
    {
        std::mutex                      mutex;
        std::map<SessionId, References> sessions;
        std::set<SessionId>             closedSessions;
        bool                            accepting{true};
        bool                            acquired{};
        Clock::time_point               started{Clock::now()};
    };

    struct Guard
    {
        std::shared_ptr<State> state;
        SessionId              session;
        bool                   active{};

        ~Guard()
        {
            if (!active) { return; }
            std::scoped_lock lock(state->mutex);
            --state->sessions.at(session).self;
        }
    };

    std::shared_ptr<State> _state{std::make_shared<State>()};

    public:
    Lease AcquireSelf(SessionId session = NoSession, bool control = false)
    {
        auto guard     = std::make_shared<Guard>();
        guard->state   = _state;
        guard->session = session;
        std::scoped_lock lock(_state->mutex);
        if (!_state->accepting && !control) { return {}; }
        if (session != NoSession && _state->closedSessions.contains(session)) { return {}; }
        ++_state->sessions[session].self;
        guard->active = true;
        return guard;
    }

    void AcquireLocal(SessionId session, std::string const& token, pid_t pid)
    {
        auto const identity = ProcessIdentity::Read(pid);
        if (!identity) { throw std::runtime_error("Checkpoint client process is not alive"); }
        std::scoped_lock lock(_state->mutex);
        if (!_state->accepting) { throw std::runtime_error("Checkpoint daemon is stopping"); }
        auto& refs = _state->sessions[session];
        if (refs.releasedLocal.contains(token)) { throw std::runtime_error("Checkpoint reference is already released"); }
        auto const [it, inserted] = refs.local.emplace(token, *identity);
        if (!inserted && it->second != *identity) { throw std::runtime_error("Checkpoint reference token has a different owner"); }
        _state->closedSessions.erase(session);
        _state->acquired = true;
    }

    void ReleaseLocal(SessionId session, std::string const& token)
    {
        std::scoped_lock lock(_state->mutex);
        auto&            refs = _state->sessions[session];
        refs.releasedLocal.insert(token);
        refs.local.erase(token);
    }

    void AcquireRemote(SessionId session, std::string const& token, std::string const& peer)
    {
        std::scoped_lock lock(_state->mutex);
        if (!_state->accepting) { throw std::runtime_error("Checkpoint daemon is stopping"); }
        auto& refs = _state->sessions[session];
        if (refs.releasedRemote.contains(token)) { throw std::runtime_error("Checkpoint reference is already released"); }
        auto const [it, inserted] = refs.remote.emplace(token, peer);
        if (!inserted && it->second != peer) { throw std::runtime_error("Checkpoint reference token has a different owner"); }
        _state->closedSessions.erase(session);
        _state->acquired = true;
    }

    void ReleaseRemote(SessionId session, std::string const& token)
    {
        std::scoped_lock lock(_state->mutex);
        auto&            refs = _state->sessions[session];
        refs.releasedRemote.insert(token);
        refs.remote.erase(token);
    }

    bool HasUnassignedWork() const
    {
        std::scoped_lock lock(_state->mutex);
        auto const       it = _state->sessions.find(NoSession);
        return it != _state->sessions.end() && it->second.self != 0;
    }

    bool IsAccepting() const
    {
        std::scoped_lock lock(_state->mutex);
        return _state->accepting;
    }

    bool TryCloseSession(SessionId session)
    {
        std::scoped_lock lock(_state->mutex);
        auto const       unassigned = _state->sessions.find(NoSession);
        if (unassigned != _state->sessions.end() && unassigned->second.self != 0) { return false; }
        auto const it = _state->sessions.find(session);
        if (it != _state->sessions.end())
        {
            auto const& refs = it->second;
            if (refs.self != 0 || !refs.local.empty() || !refs.remote.empty()) { return false; }
        }
        _state->closedSessions.insert(session);
        return true;
    }

    template <typename TPing> bool HasValidReferences(TPing const& ping, std::optional<SessionId> session = std::nullopt)
    {
        while (true)
        {
            std::vector<std::tuple<SessionId, std::string, std::string>> remotes;
            {
                std::scoped_lock lock(_state->mutex);
                for (auto const& [id, refs] : _state->sessions)
                {
                    if ((!session || id == *session) && refs.self != 0) { return true; }
                }
                for (auto& [id, refs] : _state->sessions)
                {
                    if (session && id != *session) { continue; }
                    for (auto it = refs.local.begin(); it != refs.local.end();)
                    {
                        if (it->second.IsAlive()) { return true; }
                        refs.releasedLocal.insert(it->first);
                        it = refs.local.erase(it);
                    }
                }
                for (auto const& [id, refs] : _state->sessions)
                {
                    if (session && id != *session) { continue; }
                    for (auto const& [token, peer] : refs.remote) { remotes.emplace_back(id, token, peer); }
                }
                if (remotes.empty()) { return false; }
            }
            for (auto const& [id, token, peer] : remotes)
            {
                bool const       alive = ping(peer);
                std::scoped_lock lock(_state->mutex);
                auto&            refs = _state->sessions.at(id).remote;
                auto             it   = refs.find(token);
                if (it == refs.end() || it->second != peer) { continue; }
                if (alive) { return true; }
                _state->sessions.at(id).releasedRemote.insert(token);
                refs.erase(it);
            }
        }
    }

    bool TryRetire(Clock::time_point now = Clock::now())
    {
        static constexpr auto StartupGrace = std::chrono::seconds{5};
        std::scoped_lock      lock(_state->mutex);
        if (!_state->acquired && now - _state->started < StartupGrace) { return false; }
        for (auto const& [id, refs] : _state->sessions)
        {
            static_cast<void>(id);
            if (refs.self != 0 || !refs.local.empty() || !refs.remote.empty()) { return false; }
        }
        _state->accepting = false;
        return true;
    }
};
}    // namespace checkpoint
