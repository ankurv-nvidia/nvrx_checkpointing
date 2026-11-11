#pragma once
#include "CommonMacros.h"
#include "messages.h"

#include <boost/beast/http/status.hpp>
#include <boost/beast/http/verb.hpp>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace checkpoint::service
{
using Clock = std::chrono::steady_clock;

// Reference kinds, listed in the order a validity walk visits them.
enum class Kinds : uint8_t
{
    Self   = 1,
    Pid    = 2,
    Remote = 4,
    All    = 7,
};

struct Probes
{
    std::function<bool(int64_t pid)>                processAlive;
    std::function<bool(std::string const& address)> peerAlive;
    std::function<bool(boost::beast::http::verb, std::string const& address, std::string const& target, std::string const& body)> notify;
};

[[nodiscard]] Probes  DefaultProbes();
[[nodiscard]] bool    ProcessAlive(int64_t pid);
[[nodiscard]] int64_t CurrentPid();

class Registry;

class SelfReference
{
    public:
    SelfReference() = default;
    ~SelfReference();
    SelfReference(SelfReference&& other) noexcept;
    SelfReference& operator=(SelfReference&& other) noexcept;
    SelfReference(SelfReference const&)            = delete;
    SelfReference& operator=(SelfReference const&) = delete;

    private:
    friend class Registry;
    SelfReference(Registry* registry LFTBND, size_t* count LFTBND) : _registry(registry), _count(count) {}

    Registry* _registry{};
    size_t*   _count{};
};

class SessionReferences
{
    public:
    struct State;

    SessionReferences() = default;
    ~SessionReferences();
    SessionReferences(SessionReferences&& other) noexcept;
    SessionReferences& operator=(SessionReferences&& other) noexcept;
    SessionReferences(SessionReferences const&)            = delete;
    SessionReferences& operator=(SessionReferences const&) = delete;

    private:
    friend class Registry;
    SessionReferences(Registry* registry LFTBND, std::shared_ptr<State> state) : _registry(registry), _state(std::move(state)) {}

    Registry*              _registry{};
    std::shared_ptr<State> _state;
};

// Tracks what keeps a checkpoint service alive and reaps references whose holders are gone.
class Registry
{
    public:
    explicit Registry(Clock::duration scanInterval = ConfiguredScanInterval(), Probes probes = DefaultProbes());
    ~Registry();
    CLASS_DELETE_COPY_AND_MOVE(Registry);

    [[nodiscard]] SelfReference AcquireSelf(std::string_view context) LFTBND;

    // Registers the session's pending work and, given a coordinator, holds a reference there while the session has references.
    [[nodiscard]] SessionReferences AttachSession(checkpoint::ipc::SessionId   session,
                                                  std::function<std::string()> pendingWork,
                                                  std::string                  coordinator = {},
                                                  std::string                  address     = {}) LFTBND;

    void AcquirePid(int64_t pid, checkpoint::ipc::SessionId session);
    void ReleasePid(int64_t pid, checkpoint::ipc::SessionId session);
    void ForgetPid(int64_t pid);
    void AcquireRemote(std::string const& address, checkpoint::ipc::SessionId session);
    void ReleaseRemote(std::string const& address, checkpoint::ipc::SessionId session);

    [[nodiscard]] boost::beast::http::status Handle(boost::beast::http::verb method, std::string_view path, std::string_view body);

    [[nodiscard]] bool            HasValidReference(Kinds kinds = Kinds::All);
    [[nodiscard]] bool            WaitUntilDrained(Clock::time_point deadline, Kinds kinds = Kinds::All);
    [[nodiscard]] Clock::duration DrainedFor();
    [[nodiscard]] std::string     Census();

    [[nodiscard]] static Clock::duration ConfiguredScanInterval();

    private:
    friend class SelfReference;
    friend class SessionReferences;

    struct PidEntry
    {
        size_t   count{};
        uint64_t generation{};
    };

    struct RemoteEntry
    {
        size_t            count{};
        uint64_t          generation{};
        uint32_t          failures{};
        bool              probing{};
        Clock::time_point lastProbe;
    };

    using PidKey    = std::pair<int64_t, checkpoint::ipc::SessionId>;
    using RemoteKey = std::pair<std::string, checkpoint::ipc::SessionId>;

    void ReleaseSelf_(size_t* count);
    void Detach_(std::shared_ptr<SessionReferences::State> const& state);
    void Changed_(bool wakeReaper);

    bool Walk_(Kinds kinds, std::optional<checkpoint::ipc::SessionId> session);
    bool SelfValid_(std::optional<checkpoint::ipc::SessionId> session);
    bool PidsValid_(std::optional<checkpoint::ipc::SessionId> session);
    bool RemotesValid_(std::optional<checkpoint::ipc::SessionId> session);
    void Reconcile_();
    void Scan_();
    void Run_();

    Clock::duration const _scanInterval;
    Probes const          _probes;

    std::mutex              _mutex;
    std::condition_variable _wake;
    std::condition_variable _changed;

    std::map<std::string, size_t, std::less<>>             _selfCounts;
    std::map<PidKey, PidEntry>                             _pids;
    std::map<RemoteKey, RemoteEntry>                       _remotes;
    std::vector<std::shared_ptr<SessionReferences::State>> _sessions;

    uint64_t                         _generation{};
    uint64_t                         _releases{};
    bool                             _armed{};
    bool                             _poked{};
    bool                             _stopping{};
    std::optional<Clock::time_point> _drainedSince;

    std::thread _reaper;
};

// Holds a pid reference on the local service at the port until released; null when that service does not track references.
[[nodiscard]] std::shared_ptr<void> AcquireClientReference(uint16_t port, checkpoint::ipc::SessionId session);

void WaitForService(uint16_t port, Clock::duration timeout);
}    // namespace checkpoint::service
