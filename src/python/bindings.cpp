
#include "binding.h"
#include "checkpoint.h"
#include "state.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <cstring>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

namespace nb   = nanobind;
namespace cipc = checkpoint::ipc;

static std::string StatePath(StateTree const& tree, StateNode::Handle parent, std::string_view name)
{
    auto const parentPath = tree.GetNodeId(parent);
    return parentPath.empty() ? std::string(name) : fmt::format("{}.{}", parentPath, name);
}
namespace
{
struct PythonStateTree : PyStateDictAccessor
{
    explicit PythonStateTree(RankTopology topology) : _tree(std::move(topology)) {}

    ~PythonStateTree() override
    {
        if (Py_IsInitialized() != 0)
        {
            {
                nb::gil_scoped_release release;
                try
                {
                    WaitForAsyncCalls();
                } catch (std::exception const& ex)
                {
                    fmt::print(stderr, "Unhandled asynchronous state registration error: {}\n", ex.what());
                }
            }
            OnSnapshotComplete_();
        }
        else
        {
            for (auto& object : _tensorRefObjects) { static_cast<void>(object.second.release()); }
            static_cast<void>(_dictToScan.release());
        }
    }

    CLASS_DELETE_COPY_AND_MOVE(PythonStateTree);

    StateNode& AddDictionaryChild(StateNode::Handle parent, std::string_view name) override;
    StateNode& RegisterState(StateNode::Handle parent, std::string_view name, StateNode::Variant data) override;
    StateNode& RegisterStateByFQN(std::string_view fqn, StateNode::Variant data) override;
    void       AcquireTensorObjectRefForSnapshot(StateNode::Handle node, nb::object const& obj) override;

    [[nodiscard]] std::string GetNodeId(StateNode::Handle node) const override;

    void Scan(nb::dict stateDict);
    void SaveCheckpoint(cipc::SnapshotId iteration, std::filesystem::path const& dirpath);

    void FinalizeRegistry()
    {
        ChainAsyncCall([this] {
            FinalizeStateRegistry(_tree);
            OnSnapshotComplete_();
        });
    }

    template <typename TCall> void ChainAsyncCall(TCall&& call)
    {
        std::scoped_lock lock(_asyncMutex);
        _async = std::async(std::launch::async,
                            [previous = std::move(_async), next = std::function<void()>(std::forward<TCall>(call))]() mutable {
                                try
                                {
                                    if (previous.valid()) { previous.get(); }
                                    next();
                                } catch (...)
                                {
                                    nb::gil_scoped_acquire acquire;
                                    next = {};
                                    throw;
                                }
                                {
                                    nb::gil_scoped_acquire acquire;
                                    next = {};
                                }
                            });
    }

    void WaitForAsyncCalls()
    {
        std::scoped_lock lock(_asyncMutex);
        if (_async.valid()) { _async.get(); }
    }

    [[nodiscard]] StateTree const& Tree() const { return _tree; }
    void                           SetStartedAt(int64_t startedAt) { _tree.startedAt = std::chrono::nanoseconds{startedAt}; }

    private:
    StateNode& MatchStateChild_(StateNode::Handle parent, std::string_view name, StateNode::Variant const& data);
    void       ClearTensorHandles_();
    void       ValidateTensorHandles_() const;
    void       ReleasePythonObjects_();

    void OnSnapshotComplete_()
    {
        ClearTensorHandles_();
        {
            nb::gil_scoped_acquire acquire;
            ReleasePythonObjects_();
        }
    }

    StateTree _tree;

    nb::dict _dictToScan;

    std::unordered_map<StateNode::Handle, nb::object> _tensorRefObjects;

    std::mutex        _asyncMutex;
    std::future<void> _async;
};
}    // namespace

StateNode& PythonStateTree::MatchStateChild_(StateNode::Handle parent, std::string_view name, StateNode::Variant const& data)
{
    auto& parentData = _tree.GetNode(parent).data;
    if (parentData.index() != static_cast<size_t>(StateNode::Type::Dict)) { throw std::runtime_error("Node data is not of type Dict"); }
    StateNode* namedChild          = nullptr;
    StateNode* matchingTensorChild = nullptr;
    for (auto const childHandle : std::get<StateNode::Dict>(parentData))
    {
        auto& child = _tree.GetNode(childHandle);
        if (child.name != name) { continue; }
        if (namedChild == nullptr) { namedChild = &child; }
        if (data.index() != static_cast<size_t>(StateNode::Type::TensorShard)
            || child.data.index() != static_cast<size_t>(StateNode::Type::TensorShard))
        {
            return child;
        }
        auto const& previous = std::get<StateNode::TensorShard>(child.data);
        auto const& current  = std::get<StateNode::TensorShard>(data);
        if (current.dtype == previous.dtype && current.device == previous.device && current.fulldim == previous.fulldim
            && current.sharddim == previous.sharddim)
        {
            if (matchingTensorChild == nullptr) { matchingTensorChild = &child; }
            if (previous.handle == 0) { return child; }
        }
    }
    if (matchingTensorChild != nullptr) { return *matchingTensorChild; }
    if (namedChild != nullptr) { return *namedChild; }
    throw std::runtime_error(fmt::format("supplied state '{}' is not registered", StatePath(_tree, parent, name)));
}

StateNode& PythonStateTree::AddDictionaryChild(StateNode::Handle parent, std::string_view name)
{
    if (_tree.IsFinalized())
    {
        auto& child = _tree.FindChildNode(parent, name);
        if (child.data.index() != static_cast<size_t>(StateNode::Type::Dict))
        {
            throw std::runtime_error(fmt::format("supplied state '{}' changed container type", _tree.GetNodeId(child.handle)));
        }
        return child;
    }
    auto& child = _tree.AddChild(parent, name);
    child.data  = StateNode::Dict{};
    return child;
}

StateNode& PythonStateTree::RegisterStateByFQN(std::string_view fqn, StateNode::Variant data)
{
    auto [name, parent] = [&]() -> std::pair<std::string_view, StateNode::Handle> {
        if (!_tree.IsFinalized()) { return _tree.FindOrAddNodeByFQN(fqn); }
        return _tree.FindParentDictNodeByFQN(fqn);
    }();
    return RegisterState(parent, name, std::move(data));
}

StateNode& PythonStateTree::RegisterState(StateNode::Handle parent, std::string_view stateName, StateNode::Variant data)
{
    if (!_tree.IsFinalized())
    {
        auto& child = _tree.AddChild(parent, stateName);
        child.data  = std::move(data);
        return child;
    }

    auto& child = MatchStateChild_(parent, stateName, data);
    if (child.data.index() != data.index())
    {
        throw std::runtime_error(fmt::format("supplied state '{}' changed registered type", _tree.GetNodeId(child.handle)));
    }
    auto const path = _tree.GetNodeId(child.handle);
    if (data.index() == static_cast<size_t>(StateNode::Type::Scalar))
    {
        auto const previous = std::get<StateNode::Scalar>(child.data);
        auto const current  = std::get<StateNode::Scalar>(data);
        if (current.GetType() != previous.GetType())
        {
            throw std::runtime_error(fmt::format("State '{}' changed scalar type after registry synchronization", path));
        }
    }
    else if (data.index() == static_cast<size_t>(StateNode::Type::Blob))
    {
        auto const currentSize    = std::get<StateNode::Blob>(data).size();
        auto const registeredSize = std::get<StateNode::Blob>(child.data).size();
        if (currentSize != registeredSize)
        {
            throw std::runtime_error(fmt::format(
                "State '{}' changed serialized size after registry synchronization: {} != {} bytes", path, currentSize, registeredSize));
        }
    }
    else if (data.index() == static_cast<size_t>(StateNode::Type::TensorShard))
    {
        auto const& previous = std::get<StateNode::TensorShard>(child.data);
        auto const& current  = std::get<StateNode::TensorShard>(data);
        if (current.dtype != previous.dtype || current.device != previous.device || current.fulldim != previous.fulldim
            || current.sharddim != previous.sharddim)
        {
            throw std::runtime_error(fmt::format("State '{}' changed tensor metadata after registry synchronization", path));
        }
    }
    child.data = std::move(data);
    return child;
}

void PythonStateTree::AcquireTensorObjectRefForSnapshot(StateNode::Handle node, nb::object const& obj)
{ _tensorRefObjects.insert_or_assign(node, obj); }

std::string PythonStateTree::GetNodeId(StateNode::Handle node) const
{ return _tree.GetNodeId(node); }

void PythonStateTree::ClearTensorHandles_()
{
    _tree.WalkNodes([](StateNode& node) {
        if (node.data.index() == static_cast<size_t>(StateNode::Type::TensorShard))
        {
            std::get<StateNode::TensorShard>(node.data).handle = 0;
        }
    });
}

void PythonStateTree::ValidateTensorHandles_() const
{
    _tree.WalkNodes([this](StateNode const& node) {
        if (node.data.index() == static_cast<size_t>(StateNode::Type::TensorShard)
            && std::get<StateNode::TensorShard>(node.data).handle == 0)
        {
            throw std::runtime_error(
                fmt::format("registered tensor '{}' was not supplied for this snapshot", _tree.GetNodeId(node.handle)));
        }
    });
}

void PythonStateTree::ReleasePythonObjects_()
{
    _tensorRefObjects.clear();
    _dictToScan.reset();
}

void PythonStateTree::Scan(nb::dict stateDict)
{
    ClearTensorHandles_();
    {
        nb::gil_scoped_acquire acquire;
        ReleasePythonObjects_();
        _dictToScan = std::move(stateDict);
    }
    std::unordered_set<PyObject*> visited;
    try
    {
        RegisterStateDictRecr(*this, StateNode::Handle::Root, _dictToScan, true, visited);
    } catch (nb::python_error const& error)
    {
        auto const message = std::string(error.what());
        OnSnapshotComplete_();
        throw std::runtime_error(message);
    } catch (...)
    {
        OnSnapshotComplete_();
        throw;
    }
}

void PythonStateTree::SaveCheckpoint(cipc::SnapshotId iteration, std::filesystem::path const& dirpath)
{
    try
    {
        FinalizeStateRegistry(_tree);
        ValidateTensorHandles_();
        ::SaveCheckpoint(_tree, iteration, dirpath);
    } catch (...)
    {
        OnSnapshotComplete_();
        throw;
    }
    OnSnapshotComplete_();
}

static StateNode::Scalar ScalarState(nb::handle obj)
{
    if (nb::isinstance<int64_t>(obj)) { return StateNode::Scalar{nb::cast<int64_t>(obj)}; }
    if (nb::isinstance<double>(obj)) { return StateNode::Scalar{nb::cast<double>(obj)}; }
    if (nb::isinstance<bool>(obj)) { return StateNode::Scalar{nb::cast<bool>(obj)}; }
    throw std::out_of_range("Conversion of nanobind::object to Scalar failed");
}

static StateNode::Blob SliceBlob(nb::handle value)
{
    struct SliceData
    {
        int64_t start{};
        int64_t stop{};
        int64_t step{};
    } data{};
    auto obj                      = nb::cast<nb::slice>(value);
    auto [start, stop, step, len] = obj.compute(std::numeric_limits<long>::max());
    data.start                    = start;
    data.stop                     = stop;
    data.step                     = step;
    auto const* begin             = reinterpret_cast<char const*>(&data);    // NOLINT
    return {begin, begin + sizeof(data)};
}

static StateNode::Blob StringBlob(nb::handle value)
{
    auto       string = nb::cast<nb::str>(value);
    Py_ssize_t len    = 0;
    auto*      ptr    = PyUnicode_AsWideCharString(string.ptr(), &len);
    if (ptr == nullptr) { throw nb::python_error(); }
    auto const*     begin = reinterpret_cast<char const*>(ptr);    // NOLINT
    StateNode::Blob blob(begin, begin + static_cast<size_t>(len * 2));
    PyMem_Free(ptr);
    return blob;
}

static StateNode::Blob BytesBlob(nb::handle value)
{
    auto const bytes = nb::cast<nb::bytes>(value);
    return {bytes.c_str(), bytes.c_str() + bytes.size()};
}

template <typename T, typename TPy> static StateNode::Blob TrivialTupleBlob(nb::handle value)
{
    auto            obj = nb::cast<TPy>(value);
    StateNode::Blob data(sizeof(T) * obj.size());
    auto*           ptr = reinterpret_cast<T*>(data.data());                     // NOLINT
    for (size_t i = 0; i < obj.size(); ++i) { ptr[i] = nb::cast<T>(obj[i]); }    // NOLINT
    return data;
}

template <typename TPy> static StateNode::Blob StringTupleBlob(nb::handle value)
{
    auto   obj  = nb::cast<TPy>(value);
    size_t size = 0;
    for (size_t i = 0; i < obj.size(); ++i) { size += (PyObject_Length(obj[i].ptr()) * sizeof(wchar_t)); }
    StateNode::Blob data(size);
    auto            it = data.begin();
    for (size_t i = 0; i < obj.size(); ++i)
    {
        Py_ssize_t len  = 0;
        auto*      wide = PyUnicode_AsWideCharString(obj[i].ptr(), &len);
        if (wide == nullptr) { throw nb::python_error(); }
        auto* ptr = reinterpret_cast<char*>(wide);          // NOLINT
        it        = std::copy(ptr, ptr + (len * 2), it);    // NOLINT
        PyMem_Free(wide);
    }
    return data;
}

static StateNode::Blob EnumListBlob(nb::list const& obj)
{
    std::vector<int64_t> values(obj.size());
    for (size_t i = 0; i < obj.size(); ++i) { values[i] = nb::cast<int64_t>(obj[i].attr("value")); }
    auto const* data = reinterpret_cast<char const*>(values.data());    // NOLINT
    return {data, data + (sizeof(int64_t) * values.size())};
}

static StateNode::Blob PythonRngStateBlob(nb::tuple const& tuple)
{
    auto            state = nb::cast<nb::tuple>(tuple[1]);
    StateNode::Blob data((state.size() + 1) * sizeof(int64_t));
    auto*           ptr = reinterpret_cast<int64_t*>(data.data());    // NOLINT
    ptr[0]              = nb::cast<int64_t>(tuple[0]);                // NOLINT
    for (size_t i = 0; i < state.size(); i++)
    {
        ptr[i + 1] = nb::cast<int64_t>(state[i]);    // NOLINT
    }
    return data;
}

static StateNode::Blob NumPyRngStateBlob(nb::tuple const& tuple)
{
    auto       name    = nb::cast<nb::str>(tuple[0]);
    auto       state   = nb::cast<nb::ndarray<uint32_t>>(tuple[1]);
    auto       state1  = nb::cast<int64_t>(tuple[2]);
    auto       state2  = nb::cast<int64_t>(tuple[3]);
    auto       state3  = nb::cast<float>(tuple[4]);
    Py_ssize_t len     = 0;
    auto*      nameptr = reinterpret_cast<char*>(PyUnicode_AsWideCharString(name.ptr(), &len));    // NOLINT
    if (nameptr == nullptr) { throw nb::python_error(); }
    auto rngNameLen = static_cast<uint16_t>(len);

    auto size
        = sizeof(rngNameLen) + (rngNameLen * sizeof(wchar_t)) + (state.size() * sizeof(uint32_t)) + (sizeof(int64_t) * 2) + sizeof(float);
    StateNode::Blob data(size);
    size_t          offset = 0;
    auto            append = [&](void const* source, size_t bytes) {
        std::memcpy(data.data() + offset, source, bytes);
        offset += bytes;
    };
    append(&rngNameLen, sizeof(rngNameLen));
    append(nameptr, rngNameLen * sizeof(wchar_t));
    append(state.data(), state.size() * sizeof(uint32_t));
    append(&state1, sizeof(state1));
    append(&state2, sizeof(state2));
    append(&state3, sizeof(state3));

    assert(offset == size);
    PyMem_Free(nameptr);
    return data;
}

static bool IsEnum(nb::handle const& obj)
{
    static auto Enum = nb::module_::import_("enum").attr("Enum");
    return nb::isinstance(obj, Enum);
}

static bool IsEnumList(nb::list const& obj)
{
    return std::ranges::all_of(obj, [](auto const& item) { return IsEnum(item) && nb::isinstance<int64_t>(item.attr("value")); });
}

static bool IsSimpleNamespace(nb::handle obj)
{
    static auto Namespace = nb::module_::import_("types").attr("SimpleNamespace");
    return nb::isinstance(obj, Namespace);
}

static bool IsArgParseNamespace(nb::handle obj)
{
    static auto Namespace = nb::module_::import_("argparse").attr("Namespace");
    return nb::isinstance(obj, Namespace);
}

static bool IsNamespace(nb::handle obj)
{ return IsSimpleNamespace(obj) || IsArgParseNamespace(obj); }

static bool IsPythonRngState(nb::tuple const& tupleObj)
{
    if (tupleObj.size() != 3) { return false; }
    if (!nb::isinstance<nb::int_>(tupleObj[0])) { return false; }
    if (!nb::isinstance<nb::tuple>(tupleObj[1])) { return false; }
    if (!tupleObj[2].is_none()) { return false; }
    auto stateTuple = nb::cast<nb::tuple>(tupleObj[1]);
    return std::ranges::all_of(stateTuple, [](auto&& i) { return nb::isinstance<nb::int_>(i); });
}

static bool IsNumPyRngState(nb::tuple const& tupleObj)
{
    static constexpr size_t NumpyTupleSize = 5;
    if (tupleObj.size() != NumpyTupleSize) { return false; }
    if (!nb::isinstance<nb::str>(tupleObj[0])) { return false; }
    if (!nb::isinstance<nb::ndarray<uint32_t>>(tupleObj[1])) { return false; }
    if (!nb::isinstance<nb::int_>(tupleObj[2])) { return false; }
    if (!nb::isinstance<nb::int_>(tupleObj[3])) { return false; }
    if (!nb::isinstance<nb::float_>(tupleObj[4])) { return false; }
    return true;
}

static void RegisterStateDictRecrContainerItem(PyStateDictAccessor&           scan,    // NOLINT(readability-function-cognitive-complexity)
                                               StateNode::Handle              node,
                                               nb::dict const&                container,
                                               bool                           recursive,
                                               std::unordered_set<PyObject*>& visited,
                                               nb::str const&                 key,
                                               nb::handle const&              val)
{
    // check if nonetype
    if (val.is_none()) { return; }
    std::string_view keyView = StrView(key);
    if (!nb::isinstance<nb::str>(key) && !nb::isinstance<int64_t>(key))
    {
        throw std::runtime_error(
            fmt::format("[TODO]StateDict::Register] {} key = {} is not string or int\n", scan.GetNodeId(node), keyView));
    }

    if (nb::isinstance<int64_t>(val) || nb::isinstance<double>(val) || nb::isinstance<bool>(val))
    {    //
        scan.RegisterState(node, keyView, ScalarState(val));
    }
    else if (nb::isinstance<nb::str>(val)) { scan.RegisterState(node, keyView, StringBlob(val)); }
    else if (nb::isinstance<nb::bytes>(val)) { scan.RegisterState(node, keyView, BytesBlob(val)); }
    else if (IsEnum(val))
    {
        if (!nb::isinstance<int64_t>(val.attr("value")))
        {
            throw std::runtime_error("[TODO][StateDict::Register] Only Integer enums supported");
        }
        scan.RegisterState(node, keyView, StateNode::Scalar{nb::cast<int64_t>(val.attr("value"))});
    }
    else if (nb::isinstance<nb::tuple>(val))
    {
        auto tupleSize = nb::cast<nb::tuple>(val).size();
        auto tupleObj  = nb::cast<nb::tuple>(val);
        if (tupleSize == 0) { return; }
        if (IsPythonRngState(tupleObj)) { scan.RegisterState(node, keyView, PythonRngStateBlob(tupleObj)); }
        else if (IsNumPyRngState(tupleObj)) { scan.RegisterState(node, keyView, NumPyRngStateBlob(tupleObj)); }
        else if (IsTuple<nb::str, nb::tuple>(tupleObj)) { scan.RegisterState(node, keyView, StringTupleBlob<nb::tuple>(tupleObj)); }
        else if (IsTuple<int64_t, nb::tuple>(tupleObj))
        {
            scan.RegisterState(node, keyView, TrivialTupleBlob<int64_t, nb::tuple>(tupleObj));
        }
        else if (IsTuple<double, nb::tuple>(tupleObj)) { scan.RegisterState(node, keyView, TrivialTupleBlob<double, nb::tuple>(tupleObj)); }
        else
        {
            throw std::runtime_error(
                fmt::format("[TODO]StateDict::Register] Unsupported tuple element type for key: {} obj = {}", key, nb::str(tupleObj)));
        }
    }
    else if (nb::isinstance<nb::slice>(val)) { scan.RegisterState(node, keyView, SliceBlob(val)); }
    else if (nb::isinstance<nb::list>(val))
    {
        auto listObj = nb::cast<nb::list>(val);
        if (listObj.empty()) { return; }
        // Post-`apply_factories` ShardedTensor lists land here — register each
        // element under its own FQN before the generic per-element-type checks
        // below get a chance to mis-classify them (a list of ShardedTensor
        // instances trips `Unsupported list element type`).
        if (TryRegisterTorchStateDictList(scan, node, key, listObj)) { return; }
        if (IsTuple<nb::str, nb::list>(listObj)) { scan.RegisterState(node, keyView, StringTupleBlob<nb::list>(listObj)); }
        else if (IsTuple<int64_t, nb::list>(listObj)) { scan.RegisterState(node, keyView, TrivialTupleBlob<int64_t, nb::list>(listObj)); }
        else if (IsTuple<double, nb::list>(listObj)) { scan.RegisterState(node, keyView, TrivialTupleBlob<double, nb::list>(listObj)); }
        else if (IsEnumList(listObj)) { scan.RegisterState(node, keyView, EnumListBlob(listObj)); }
        else if (IsTuple<nb::dict, nb::list>(listObj))
        {
            if (recursive)
            {
                for (size_t i = 0; i < listObj.size(); ++i)
                {
                    auto                   item      = listObj[i];
                    auto&                  child     = scan.AddDictionaryChild(node, fmt::format("{}.[list]#{}", keyView, i));
                    auto                   childDict = nb::cast<nb::dict>(item);
                    nb::gil_scoped_release release;
                    RegisterStateDictRecr(scan, child.handle, childDict, recursive, visited);
                }
            }
        }
        else if (IsTuple<nb::list, nb::list>(listObj))
        {
            if (recursive)
            {
                for (size_t i = 0; i < listObj.size(); ++i)
                {
                    auto item = nb::cast<nb::list>(listObj[i]);
                    for (size_t j = 0; j < item.size(); ++j)
                    {
                        auto  item1 = item[j];
                        auto& child = scan.AddDictionaryChild(node, fmt::format("{}.[list]#{}.[list]#{}", key, i, j));
                        if (!nb::isinstance<nb::dict>(item1))
                        {
                            throw std::runtime_error(
                                fmt::format("[TODO]StateDict::Register] Unsupported list element type for {}. Expected dict, got {}",
                                            child.name,
                                            nb::str(item1.type())));
                        }
                        auto                   childDict = nb::cast<nb::dict>(item1);
                        nb::gil_scoped_release release;
                        RegisterStateDictRecr(scan, child.handle, childDict, recursive, visited);
                    }
                }
            }
        }
        else
        {
            throw std::runtime_error(
                fmt::format("[TODO]StateDict::Register] Unsupported list element type for key: {} obj = {}", key, nb::str(listObj)));
        }
    }
    else if (TryRegisterTorchStateDict(scan, node, container, key, val)) {}
    else if (recursive)
    {
        if (nb::isinstance<nb::dict>(val))
        {
            auto&                  child     = scan.AddDictionaryChild(node, keyView);
            auto                   childDict = nb::cast<nb::dict>(val);
            nb::gil_scoped_release release;
            RegisterStateDictRecr(scan, child.handle, childDict, recursive, visited);
        }
        // if namespace
        else if (IsNamespace(val))
        {
            auto&                  child     = scan.AddDictionaryChild(node, keyView);
            auto                   childDict = nb::cast<nb::dict>(val.attr("__dict__"));
            nb::gil_scoped_release release;
            RegisterStateDictRecr(scan, child.handle, childDict, recursive, visited);
        }
    }
    else
    {
        fmt::print(stderr,
                   "[native_ckpt:cli][TODO][StateDict::Register] Skipping unsupported type for key: {} type={}\n",
                   keyView,
                   nb::str(val.type()));
    }
}

void RegisterStateDictRecr(PyStateDictAccessor&           scan,
                           StateNode::Handle              node,
                           nb::dict const&                container,
                           bool                           recursive,
                           std::unordered_set<PyObject*>& visited)
{
    if (visited.contains(container.ptr())) { return; }
    visited.insert(container.ptr());

    std::optional<nb::list> keys;
    size_t                  keyCount = 0;
    {
        nb::gil_scoped_acquire acquire;
        keys.emplace(container.attr("keys")());
        keyCount = keys->size();
    }
    try
    {
        for (size_t index = 0; index < keyCount; ++index)
        {
            nb::gil_scoped_acquire acquire;
            auto                   key   = nb::borrow<nb::object>((*keys)[index]);
            auto                   value = nb::borrow<nb::object>(container[key]);
            try
            {
                RegisterStateDictRecrContainerItem(scan, node, container, recursive, visited, nb::str(key), value);
            } catch (std::exception const& e)
            {
                fmt::print(stderr,
                           "[native_ckpt:cli][ERROR] Error processing prefix = {}, key: {}: {}\n",
                           scan.GetNodeId(node),
                           nb::str(key),
                           e.what());
                throw;
            }
        }
    } catch (...)
    {
        nb::gil_scoped_acquire acquire;
        keys.reset();
        throw;
    }
    {
        nb::gil_scoped_acquire acquire;
        keys.reset();
    }
}

static void RegisterStateDict(PythonStateTree& self, nb::dict const& stateDict)
{
    self.ChainAsyncCall([&self, stateDict = nb::dict(stateDict)]() mutable { self.Scan(std::move(stateDict)); });
}

static void SaveAsync(PythonStateTree& self, int iteration, std::filesystem::path const& dirpath)
{
    self.ChainAsyncCall([&self, iteration, dirpath] { self.SaveCheckpoint(static_cast<cipc::SnapshotId>(iteration), dirpath); });
}

NB_MODULE(checkpoint, m)    // NOLINT
{
    m.def("ping", &SvcPing);
    m.def("svc_stop", &SvcStop);
    m.def("svc_spindown", &SvcSpinDown);
    m.def(
        "wait_for_ckpt_finalize",
        [](uint32_t sessionId, int64_t iteration) {
            nb::gil_scoped_release release;
            WaitForCkptFinalize(sessionId, iteration);
        },
        nb::arg("session_id"),
        nb::arg("iteration"));
    auto cls = nb::class_<PythonStateTree>(std::move(m), "model_state");

    cls.def_static(
           "create_root",
           [](uint32_t globalRank, uint32_t globalWorldSize, std::vector<std::string> peers, uint32_t ranksPerPeer, uint32_t peersPerNode) {
               if (ranksPerPeer == 0)
               {
                   if (peers.empty())
                   {
                       throw std::runtime_error("create_root: ranks_per_peer auto-detect requires a non-empty peers list");
                   }
                   auto numPeers = static_cast<uint32_t>(peers.size());
                   ranksPerPeer  = (globalWorldSize + numPeers - 1) / numPeers;
               }
               else
               {
                   if (peers.empty())
                   {
                       if (ranksPerPeer == globalWorldSize) { peers.emplace_back("localhost"); }
                       else
                       {
                           throw std::runtime_error("create_root: peers list cannot be empty when ranks_per_peer is specified and not "
                                                    "equal to global_world_size");
                       }
                   }
                   else if (peers.size() * ranksPerPeer < globalWorldSize)
                   {
                       throw std::runtime_error(
                           fmt::format("create_root: peers list size {} is too small for global_world_size {} and ranks_per_peer {}",
                                       peers.size(),
                                       globalWorldSize,
                                       ranksPerPeer));
                   }
               }
               return std::make_unique<PythonStateTree>(RankTopology{
                   .rank         = Rank{globalRank},
                   .worldSize    = Rank{globalWorldSize},
                   .ranksPerPeer = ranksPerPeer,
                   .peersPerNode = peersPerNode,
                   .peers        = std::move(peers),
               });
           },
           nb::arg("global_rank"),
           nb::arg("global_world_size"),
           nb::arg("peers"),
           nb::arg("ranks_per_peer") = 0,
           nb::arg("peers_per_node") = 1)
        .def_prop_ro("globalRank", [](PythonStateTree const& self) { return self.Tree().GlobalRank(); })
        .def_prop_ro("globalWorldSize", [](PythonStateTree const& self) { return self.Tree().GlobalWorldSize(); })
        .def_prop_rw(
            "startedAt",
            [](PythonStateTree const& self) { return self.Tree().startedAt.count(); },
            [](PythonStateTree& self, int64_t startedAt) { self.SetStartedAt(startedAt); })
        .def("register_state_dict", &RegisterStateDict)
        .def("finalize_registry", [](PythonStateTree& self) { self.FinalizeRegistry(); })
        .def_prop_ro("snapshot_size",
                     [](PythonStateTree& self) {
                         nb::gil_scoped_release release;
                         self.WaitForAsyncCalls();
                         return self.Tree().GetTotalSnapshotStateSizeInBytes();
                     })
        .def("save_async", &SaveAsync, nb::arg("iteration"), nb::arg("dirpath"))
        .def(
            "save",
            [](PythonStateTree& self, int iteration, std::filesystem::path const& dirpath) {
                SaveAsync(self, iteration, dirpath);
                nb::gil_scoped_release release;
                self.WaitForAsyncCalls();
            },
            nb::arg("iteration"),
            nb::arg("dirpath"))
        .def("wait_for_deferred_save", [](PythonStateTree& self) {
            nb::gil_scoped_release release;
            self.WaitForAsyncCalls();
        });
}
