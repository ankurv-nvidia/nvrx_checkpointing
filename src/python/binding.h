
#include "checkpoint.h"
#include "fmttensor.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

template <typename T, typename TPy> static bool IsTuple(TPy const& tupleObj)
{
    for (size_t i = 0; i < tupleObj.size(); ++i)
    {
        if (!nanobind::isinstance<T>(tupleObj[i])) { return false; }
    }
    return true;
}

inline std::string_view StrView(nanobind::str const& s, std::string_view const& nullReplacement = {})
{
    Py_ssize_t  size = 0;
    auto const* ptr  = PyUnicode_AsUTF8AndSize(s.ptr(), &size);
    return ptr == nullptr ? nullReplacement : std::string_view{ptr, static_cast<size_t>(size)};
}

inline std::string Str(nanobind::str const& s, std::string_view const& nullReplacement = {})
{
    Py_ssize_t  size = 0;
    auto const* ptr  = PyUnicode_AsUTF8AndSize(s.ptr(), &size);
    return ptr == nullptr ? std::string(nullReplacement) : std::string{ptr, static_cast<size_t>(size)};
}

template <> struct fmt::formatter<nanobind::str> : fmt::formatter<std::string_view>
{
    auto format(nanobind::str const& obj, fmt::format_context& ctx) const    // NOLINT
    { return fmt::format_to(ctx.out(), "{}", StrView(obj, "<null>")); }
};

class PyStateDictAccessor
{
    public:
    PyStateDictAccessor()          = default;
    virtual ~PyStateDictAccessor() = default;
    CLASS_DELETE_COPY_AND_MOVE(PyStateDictAccessor);

    virtual StateNode& AddDictionaryChild(StateNode::Handle parent, std::string_view name)                     = 0;
    virtual StateNode& RegisterState(StateNode::Handle parent, std::string_view name, StateNode::Variant data) = 0;
    virtual StateNode& RegisterStateByFQN(std::string_view name, StateNode::Variant data)                      = 0;
    virtual void       AcquireTensorObjectRefForSnapshot(StateNode::Handle node, nanobind::object const& obj)  = 0;

    [[nodiscard]] virtual std::string GetNodeId(StateNode::Handle node) const = 0;
};

bool TryRegisterTorchStateDict(PyStateDictAccessor&  scan,
                               StateNode::Handle     node,
                               nanobind::dict const& container,
                               nanobind::str const&  key,
                               nanobind::handle      val);

// Detect a list-of-ShardedTensor (what `apply_factories` produces in place of a
// ShardedTensorFactory) and register every element. Returns false on any other
// list shape so the generic walker keeps owning the existing list cases.
bool TryRegisterTorchStateDictList(PyStateDictAccessor&  scan,
                                   StateNode::Handle     node,
                                   nanobind::str const&  key,
                                   nanobind::list const& listObj);

void RegisterStateDictRecr(PyStateDictAccessor&           scan,
                           StateNode::Handle              node,
                           nanobind::dict const&          container,
                           bool                           recursive,
                           std::unordered_set<PyObject*>& visited);
