
#include "binding.h"
#include "checkpoint.h"
#include "state.h"
#include "stateserdes.h"
#include "tensor.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <optional>

namespace nb   = nanobind;
namespace cipc = checkpoint::ipc;
namespace
{
struct StateTreeScalarHandler : StateTree::StateHandler<StateTree::StateType::Scalar>
{
    StateTreeScalarHandler(nb::dict containerIn, nb::str const& nameIn) :
        container(std::move(containerIn)), StateTree::StateHandler<StateTree::StateType::Scalar>(StrView(nameIn)), key(nameIn)
    {}

    void Update() override {};

    StateTree::Scalar GetScalarState() override
    {
        nanobind::gil_scoped_acquire acquire;

        auto obj = container[key];
        if (nb::isinstance<int64_t>(obj)) { return {nb::cast<int64_t>(obj)}; }
        if (nb::isinstance<double>(obj)) { return {nb::cast<double>(obj)}; }
        if (nb::isinstance<bool>(obj)) { return {nb::cast<bool>(obj)}; }
        throw std::out_of_range("Conversion of nanobind::object to Scalar failed");
    }

    void SetScalarState(StateTree::Scalar const& value) override
    {
        nanobind::gil_scoped_acquire acquire;

        if (nb::isinstance<int64_t>(container)) { container[key] = value.Cast<int64_t>(); }
        else if (nb::isinstance<double>(container)) { container[key] = value.Cast<double>(); }
        else if (nb::isinstance<bool>(container)) { container[key] = value.Cast<bool>(); }
        else
        {
            throw std::out_of_range("Conversion of Scalar to nanobind::object failed");
        }
    }

    nb::dict container;
    nb::str  key;
};

struct StateTreeEnumHandler : StateTree::StateHandler<StateTree::StateType::Scalar>
{
    StateTreeEnumHandler(nb::dict containerIn, nb::str const& nameIn) :
        container(std::move(containerIn)), StateTree::StateHandler<StateTree::StateType::Scalar>(StrView(nameIn)), key(nameIn)
    {}

    void Update() override {};

    StateTree::Scalar GetScalarState() override
    {
        nanobind::gil_scoped_acquire acquire;

        auto obj = container[key];
        auto val = obj.attr("value");
        return {nb::cast<int64_t>(val)};
    }

    void SetScalarState(StateTree::Scalar const& value) override
    {
        nanobind::gil_scoped_acquire acquire;

        if (nb::isinstance<int64_t>(container)) { container[key] = value.Cast<int64_t>(); }
        else if (nb::isinstance<double>(container)) { container[key] = value.Cast<double>(); }
        else if (nb::isinstance<bool>(container)) { container[key] = value.Cast<bool>(); }
        else
        {
            throw std::out_of_range("Conversion of Scalar to nanobind::object failed");
        }
    }

    nb::dict container;
    nb::str  key;
};

struct StateTreeSliceHandler : StateTree::StateHandler<StateTree::StateType::Blob>
{
    StateTreeSliceHandler(nb::dict containerIn, nb::str const& nameIn) :
        container(std::move(containerIn)), StateTree::StateHandler<StateTree::StateType::Blob>(StrView(nameIn)), key(nameIn)
    {}

    void Update() override {};

    StateNode::Blob GetBlob() override
    {
        // TODO_BG_SERIALIZE: Measure this GIL-held snapshot serialization and move it behind immutable capture.
        nanobind::gil_scoped_acquire acquire;

        auto obj                      = nb::cast<nb::slice>(container[key]);
        auto [start, stop, step, len] = obj.compute(std::numeric_limits<long>::max());
        data.start                    = start;
        data.stop                     = stop;
        data.step                     = step;
        return std::vector<char>(reinterpret_cast<char*>(&data), reinterpret_cast<char*>(&data) + sizeof(data));    // NOLINT
    }

    void SetBlob(std::string_view const& value) override
    {
        nanobind::gil_scoped_acquire acquire;
        throw std::runtime_error("TODO");
    }

    struct SliceData
    {
        int64_t start;
        int64_t stop;
        int64_t step;
    } data{};

    nb::dict container;
    nb::str  key;
};

struct StateTreeStringHandler : StateTree::StateHandler<StateTree::StateType::Blob>
{
    StateTreeStringHandler(nb::dict containerIn, nb::str const& nameIn) :
        container(std::move(containerIn)), StateTree::StateHandler<StateTree::StateType::Blob>(StrView(nameIn)), key(nameIn)
    {}

    void Update() override {};

    StateNode::Blob GetBlob() override
    {
        // TODO_BG_SERIALIZE: Measure this GIL-held snapshot serialization and move it behind immutable capture.
        nanobind::gil_scoped_acquire acquire;

        auto       value = nb::cast<nb::str>(container[key]);
        Py_ssize_t len   = 0;
        auto*      ptr   = PyUnicode_AsWideCharString(value.ptr(), &len);
        return std::vector<char>(reinterpret_cast<char*>(ptr), reinterpret_cast<char*>(ptr) + static_cast<size_t>(len * 2));    // NOLINT
    }

    void SetBlob(std::string_view const& value) override
    {
        nanobind::gil_scoped_acquire acquire;
        throw std::runtime_error("TODO");
        // if (nb::isinstance<nb::str>(container)) { container[key] = std::string(value); }
        // else
        // {
        //     throw std::out_of_range("Conversion of Blob to nanobind::object failed");
        // }
    }

    nb::dict container;
    nb::str  key;
};

struct StateTreeBytesHandler : StateTree::StateHandler<StateTree::StateType::Blob>
{
    StateTreeBytesHandler(nb::dict containerIn, nb::str const& nameIn) :
        container(std::move(containerIn)), StateTree::StateHandler<StateTree::StateType::Blob>(StrView(nameIn)), key(nameIn)
    {}

    void               Update() override {}
    [[nodiscard]] bool IsSnapshotScoped() const override { return true; }

    StateNode::Blob GetBlob() override
    {
        // TODO_BG_SERIALIZE: Measure this GIL-held snapshot serialization and move it behind immutable capture.
        nanobind::gil_scoped_acquire acquire;
        auto const                   value = nb::cast<nb::bytes>(container[key]);
        return {value.c_str(), value.c_str() + value.size()};
    }

    void SetBlob(std::string_view const& value) override
    {
        nanobind::gil_scoped_acquire acquire;
        container[key] = nb::bytes(value.data(), value.size());
    }

    nb::dict container;
    nb::str  key;
};

template <typename T, typename TPy> struct StateTreeTrivialTupleHandler : StateTree::StateHandler<StateTree::StateType::Blob>
{
    StateTreeTrivialTupleHandler(nb::dict containerIn, nb::str const& nameIn) :
        container(std::move(containerIn)), StateTree::StateHandler<StateTree::StateType::Blob>(StrView(nameIn)), key(nameIn)
    {}

    void Update() override {};

    StateNode::Blob GetBlob() override
    {
        // TODO_BG_SERIALIZE: Measure this GIL-held snapshot serialization and move it behind immutable capture.
        nanobind::gil_scoped_acquire acquire;
        auto                         obj = nb::cast<TPy>(container[key]);
        std::vector<char>            data(sizeof(T) * obj.size());
        auto*                        ptr = reinterpret_cast<T*>(data.data());        // NOLINT
        for (size_t i = 0; i < obj.size(); ++i) { ptr[i] = nb::cast<T>(obj[i]); }    // NOLINT
        return data;
    }

    void SetBlob(std::string_view const& value) override
    {
        nanobind::gil_scoped_acquire acquire;
        throw std::runtime_error("TODO");
        // if (nb::isinstance<nb::str>(container)) { container[key] = std::string(value); }
        // else
        //{
        //     throw std::out_of_range("Conversion of Blob to nanobind::object failed");
        // }
    }

    nb::dict container;
    nb::str  key;
};

template <typename TPy> struct StateTreeStringTupleHandler : StateTree::StateHandler<StateTree::StateType::Blob>
{
    StateTreeStringTupleHandler(nb::dict containerIn, nb::str const& nameIn) :
        container(std::move(containerIn)), StateTree::StateHandler<StateTree::StateType::Blob>(StrView(nameIn)), key(nameIn)
    {}

    void Update() override {};

    StateNode::Blob GetBlob() override
    {
        // TODO_BG_SERIALIZE: Measure this GIL-held snapshot serialization and move it behind immutable capture.
        nanobind::gil_scoped_acquire acquire;
        auto                         obj  = nb::cast<TPy>(container[key]);
        size_t                       size = 0;
        for (size_t i = 0; i < obj.size(); ++i) { size += (PyObject_Length(obj[i].ptr()) * sizeof(wchar_t)); }
        std::vector<char> data(size);
        auto              it = data.begin();
        for (size_t i = 0; i < obj.size(); ++i)
        {
            Py_ssize_t len = 0;
            auto*      ptr = reinterpret_cast<char*>(PyUnicode_AsWideCharString(obj[i].ptr(), &len));    // NOLINT
            it             = std::copy(ptr, ptr + (len * 2), it);                                        // NOLINT
        }
        return data;
    }

    void SetBlob(std::string_view const& value) override
    {
        nanobind::gil_scoped_acquire acquire;
        throw std::runtime_error("TODO");
        // if (nb::isinstance<nb::str>(container)) { container[key] = std::string(value); }
        // else
        //{
        //     throw std::out_of_range("Conversion of Blob to nanobind::object failed");
        // }
    }

    nb::dict container;
    nb::str  key;
};

struct StateTreePythonRngStateHandler : StateTree::StateHandler<StateTree::StateType::Blob>
{
    StateTreePythonRngStateHandler(nb::dict containerIn, nb::str const& nameIn) :
        container(std::move(containerIn)), StateTree::StateHandler<StateTree::StateType::Blob>(StrView(nameIn)), key(nameIn)
    {}

    void Update() override {};

    StateNode::Blob GetBlob() override
    {
        // TODO_BG_SERIALIZE: Measure this GIL-held snapshot serialization and move it behind immutable capture.
        nanobind::gil_scoped_acquire acquire;
        auto                         tup   = nb::cast<nb::tuple>(container[key]);
        auto                         tup1  = nb::cast<nb::tuple>(tup[1]);
        size_t                       count = tup1.size() + 1;
        std::vector<char>            data(count * sizeof(int64_t));
        auto*                        ptr = reinterpret_cast<int64_t*>(data.data());    // NOLINT
        ptr[0]                           = nb::cast<int64_t>(tup[0]);                  // NOLINT
        for (size_t i = 0; i < tup1.size(); i++)
        {
            ptr[i + 1] = nb::cast<int64_t>(tup1[i]);    // NOLINT
        }
        return data;
    }

    void SetBlob(std::string_view const& value) override
    {
        nanobind::gil_scoped_acquire acquire;
        throw std::runtime_error("TODO");
        // if (nb::isinstance<nb::str>(container)) { container[key] = std::string(value); }
        // else
        //{
        //     throw std::out_of_range("Conversion of Blob to nanobind::object failed");
        // }
    }

    nb::dict container;
    nb::str  key;
};

struct StateTreeNumPyRngStateHandler : StateTree::StateHandler<StateTree::StateType::Blob>
{
    StateTreeNumPyRngStateHandler(nb::dict containerIn, nb::str const& nameIn) :
        container(std::move(containerIn)), StateTree::StateHandler<StateTree::StateType::Blob>(StrView(nameIn)), key(nameIn)
    {}

    void Update() override {};

    StateNode::Blob GetBlob() override
    {
        // TODO_BG_SERIALIZE: Measure this GIL-held snapshot serialization and move it behind immutable capture.
        nanobind::gil_scoped_acquire acquire;

        auto       tup        = nb::cast<nb::tuple>(container[key]);
        auto       name       = nb::cast<nb::str>(tup[0]);
        auto       state      = nb::cast<nb::ndarray<uint32_t>>(tup[1]);
        auto       state1     = nb::cast<int64_t>(tup[2]);
        auto       state2     = nb::cast<int64_t>(tup[3]);
        auto       state3     = nb::cast<float>(tup[4]);
        Py_ssize_t len        = 0;
        auto*      nameptr    = reinterpret_cast<char*>(PyUnicode_AsWideCharString(name.ptr(), &len));    // NOLINT
        auto       rngNameLen = static_cast<uint16_t>(len);

        auto size = sizeof(rngNameLen) + (rngNameLen * sizeof(wchar_t)) + (state.size() * sizeof(uint32_t)) + (sizeof(int64_t) * 2)
                    + sizeof(float);
        std::vector<char> data(size);

        auto it = data.begin();
        it      = std::copy(&rngNameLen, (&rngNameLen + sizeof(uint16_t)), it);                   // NOLINT
        it      = std::copy(nameptr, nameptr + (rngNameLen * sizeof(wchar_t)), it);               // NOLINT
        it      = std::copy(state.data(), state.data() + state.size() * sizeof(uint32_t), it);    // NOLINT
        it      = std::copy(&state1, &state1 + sizeof(int64_t), it);                              // NOLINT
        it      = std::copy(&state2, &state2 + sizeof(int64_t), it);                              // NOLINT
        it      = std::copy(&state3, &state3 + sizeof(float), it);                                // NOLINT

        assert(std::distance(data.begin(), it) == static_cast<std::ptrdiff_t>(size));

        return data;
    }

    void SetBlob(std::string_view const& value) override
    {
        nanobind::gil_scoped_acquire acquire;
        throw std::runtime_error("TODO");
        // if (nb::isinstance<nb::str>(container)) { container[key] = std::string(value); }
        // else
        //{
        //     throw std::out_of_range("Conversion of Blob to nanobind::object failed");
        // }
    }

    nb::dict container;
    nb::str  key;
};
}    // namespace

static bool IsEnum(nb::handle const& obj)
{
    static auto Enum = nb::module_::import_("enum").attr("Enum");
    return nb::isinstance(obj, Enum);
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

namespace
{
struct RegisteredStateDict final : StateTree::IStateHandler
{
    private:
    struct MergeState
    {
        std::unordered_set<PyObject*>   visited;
        std::unordered_set<std::string> supplied;
    };

    public:
    explicit RegisteredStateDict(nb::dict stateIn) : StateTree::IStateHandler(""), state(std::move(stateIn)) {}

    StateTree::StateType GetType() override { return StateTree::StateType::Dict; }
    void                 Update() override {}

    void Merge(nb::dict const& fresh)
    {
        MergeState mergeState;
        MergeDictionary(state, fresh, "", mergeState);
    }

    [[noreturn]] StateTree::Scalar GetScalarState() override { throw std::logic_error("registered state dictionary has no scalar value"); }
    [[noreturn]] StateNode::Blob   GetBlob() override { throw std::logic_error("registered state dictionary has no blob value"); }
    [[noreturn]] DS4Tensor         GetTensorState() override { throw std::logic_error("registered state dictionary has no tensor value"); }
    [[noreturn]] void              SetScalarState(StateTree::Scalar const&) override
    { throw std::logic_error("registered state dictionary has no scalar value"); }
    [[noreturn]] void SetBlob(std::string_view const&) override { throw std::logic_error("registered state dictionary has no blob value"); }
    [[noreturn]] void SetTensorState(DS4Tensor const&) override
    { throw std::logic_error("registered state dictionary has no tensor value"); }

    private:
    static std::string JoinStatePath(std::string_view prefix, nb::handle key)
    {
        auto const keyString = Str(nb::str(key));
        return prefix.empty() ? keyString : fmt::format("{}.{}", prefix, keyString);
    }

    static bool IsEmptySequence(nb::handle value)
    {
        if (nb::isinstance<nb::list>(value)) { return nb::cast<nb::list>(value).empty(); }
        if (nb::isinstance<nb::tuple>(value)) { return nb::cast<nb::tuple>(value).empty(); }
        return false;
    }

    static void MarkSupplied(MergeState& mergeState, std::string const& path)
    {
        if (!mergeState.supplied.insert(path).second) { throw std::runtime_error(fmt::format("duplicate normalized state '{}'", path)); }
    }

    bool MergeValue(nb::handle cached, nb::handle fresh, std::string const& path, MergeState& mergeState)
    {
        if (fresh.is_none() || IsEmptySequence(fresh)) { return false; }
        if (cached.ptr() == fresh.ptr()) { return false; }

        auto cachedDict = [&]() -> std::optional<nb::dict> {
            if (nb::isinstance<nb::dict>(cached)) { return nb::cast<nb::dict>(cached); }
            if (IsNamespace(cached)) { return nb::cast<nb::dict>(cached.attr("__dict__")); }
            return std::nullopt;
        }();
        auto freshDict = [&]() -> std::optional<nb::dict> {
            if (nb::isinstance<nb::dict>(fresh)) { return nb::cast<nb::dict>(fresh); }
            if (IsNamespace(fresh)) { return nb::cast<nb::dict>(fresh.attr("__dict__")); }
            return std::nullopt;
        }();
        if (cachedDict || freshDict)
        {
            if (!cachedDict || !freshDict) { throw std::runtime_error(fmt::format("supplied state '{}' changed container type", path)); }
            MergeDictionary(*cachedDict, *freshDict, path, mergeState);
            return false;
        }

        if (nb::isinstance<nb::list>(cached) || nb::isinstance<nb::list>(fresh))
        {
            if (!nb::isinstance<nb::list>(cached) || !nb::isinstance<nb::list>(fresh))
            {
                throw std::runtime_error(fmt::format("supplied state '{}' changed container type", path));
            }
            auto cachedList = nb::cast<nb::list>(cached);
            auto freshList  = nb::cast<nb::list>(fresh);
            if (cachedList.size() != freshList.size())
            {
                throw std::runtime_error(
                    fmt::format("supplied state '{}' changed list length from {} to {}", path, cachedList.size(), freshList.size()));
            }
            if (!mergeState.visited.insert(freshList.ptr()).second) { return false; }
            for (size_t i = 0; i < freshList.size(); ++i)
            {
                auto const itemPath = fmt::format("{}.[list]#{}", path, i);
                if (MergeValue(cachedList[i], freshList[i], itemPath, mergeState)) { cachedList[i] = freshList[i]; }
            }
            return false;
        }

        MarkSupplied(mergeState, path);
        return true;
    }

    bool MergeDottedKey(nb::dict const& cached, nb::str const& key, nb::handle fresh, std::string_view prefix, MergeState& mergeState)
    {
        auto const keyString = Str(key);
        auto       current   = cached;
        size_t     begin     = 0;
        while (true)
        {
            auto const end       = keyString.find('.', begin);
            auto const component = std::string_view(keyString).substr(begin, end == std::string::npos ? end : end - begin);
            nb::str    componentKey(component.data(), component.size());
            if (!current.contains(componentKey)) { return false; }
            if (end == std::string::npos)
            {
                auto const path = prefix.empty() ? keyString : fmt::format("{}.{}", prefix, keyString);
                if (MergeValue(current[componentKey], fresh, path, mergeState)) { current[componentKey] = fresh; }
                return true;
            }
            auto value = current[componentKey];
            if (!nb::isinstance<nb::dict>(value)) { return false; }
            current = nb::cast<nb::dict>(value);
            begin   = end + 1;
        }
    }

    void MergeDictionary(nb::dict const& cached, nb::dict const& fresh, std::string_view prefix, MergeState& mergeState)
    {
        if (cached.ptr() == fresh.ptr() || !mergeState.visited.insert(fresh.ptr()).second) { return; }
        for (auto const& item : fresh)
        {
            auto const key   = item.first;
            auto const value = item.second;
            if (cached.contains(key))
            {
                auto const path = JoinStatePath(prefix, key);
                if (MergeValue(cached[key], value, path, mergeState)) { cached[key] = value; }
                continue;
            }
            if (nb::isinstance<nb::str>(key) && MergeDottedKey(cached, nb::cast<nb::str>(key), value, prefix, mergeState)) { continue; }
            throw std::runtime_error(fmt::format("supplied state '{}' is not registered", JoinStatePath(prefix, key)));
        }
    }

    nb::dict state;
};
}    // namespace

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

bool RegisterTorchStateDict(StateTree& state, nb::handle val);

static void RegisterStateDictRecrContainerItem(StateTree&                     root,    // NOLINT(readability-function-cognitive-complexity)
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
            fmt::format("[TODO]StateDict::Register] {} key = {} is not string or int\n", root.GetNodeId(node), keyView));
    }

    if (nb::isinstance<int64_t>(val) || nb::isinstance<double>(val) || nb::isinstance<bool>(val))
    {    //
        root.RegisterStateHandler(node, std::make_unique<StateTreeScalarHandler>(container, key));
    }
    else if (nb::isinstance<nb::str>(val)) { root.RegisterStateHandler(node, std::make_unique<StateTreeStringHandler>(container, key)); }
    else if (nb::isinstance<nb::bytes>(val)) { root.RegisterStateHandler(node, std::make_unique<StateTreeBytesHandler>(container, key)); }
    else if (IsEnum(val))
    {
        if (!nb::isinstance<int64_t>(val.attr("value")))
        {
            throw std::runtime_error("[TODO][StateDict::Register] Only Integer enums supported");
        }
        root.RegisterStateHandler(node, std::make_unique<StateTreeEnumHandler>(container, key));
    }
    else if (nb::isinstance<nb::tuple>(val))
    {
        auto tupleSize = nb::cast<nb::tuple>(val).size();
        auto tupleObj  = nb::cast<nb::tuple>(val);
        if (tupleSize == 0) { return; }
        if (IsPythonRngState(tupleObj))
        {
            root.RegisterStateHandler(node, std::make_unique<StateTreePythonRngStateHandler>(container, key));
        }
        else if (IsNumPyRngState(tupleObj))
        {
            root.RegisterStateHandler(node, std::make_unique<StateTreeNumPyRngStateHandler>(container, key));
        }
        else if (IsTuple<nb::str, nb::tuple>(tupleObj))
        {
            root.RegisterStateHandler(node, std::make_unique<StateTreeStringTupleHandler<nb::tuple>>(container, key));
        }
        else if (IsTuple<int64_t, nb::tuple>(tupleObj))
        {
            root.RegisterStateHandler(node, std::make_unique<StateTreeTrivialTupleHandler<int64_t, nb::tuple>>(container, key));
        }
        else if (IsTuple<double, nb::tuple>(tupleObj))
        {
            root.RegisterStateHandler(node, std::make_unique<StateTreeTrivialTupleHandler<double, nb::tuple>>(container, key));
        }
        else
        {
            throw std::runtime_error(
                fmt::format("[TODO]StateDict::Register] Unsupported tuple element type for key: {} obj = {}", key, nb::str(tupleObj)));
        }
    }
    else if (nb::isinstance<nb::slice>(val)) { root.RegisterStateHandler(node, std::make_unique<StateTreeSliceHandler>(container, key)); }
    else if (nb::isinstance<nb::list>(val))
    {
        auto listObj = nb::cast<nb::list>(val);
        if (listObj.empty()) { return; }
        // Post-`apply_factories` ShardedTensor lists land here — register each
        // element under its own FQN before the generic per-element-type checks
        // below get a chance to mis-classify them (a list of ShardedTensor
        // instances trips `Unsupported list element type`).
        if (TryRegisterTorchStateDictList(root, node, key, listObj)) { return; }
        if (IsTuple<nb::str, nb::list>(listObj))
        {
            root.RegisterStateHandler(node, std::make_unique<StateTreeStringTupleHandler<nb::list>>(container, key));
        }
        else if (IsTuple<int64_t, nb::list>(listObj))
        {
            root.RegisterStateHandler(node, std::make_unique<StateTreeTrivialTupleHandler<int64_t, nb::list>>(container, key));
        }
        else if (IsTuple<double, nb::list>(listObj))
        {
            root.RegisterStateHandler(node, std::make_unique<StateTreeTrivialTupleHandler<double, nb::list>>(container, key));
        }
        else if (IsTuple<nb::dict, nb::list>(listObj))
        {
            if (recursive)
            {
                for (size_t i = 0; i < listObj.size(); ++i)
                {
                    auto  item  = listObj[i];
                    auto& child = root.AddChild(node, fmt::format("{}.[list]#{}", keyView, i));
                    child.data  = StateNode::Dict{};
                    // nanobind::gil_scoped_release release;
                    RegisterStateDictRecr(root, child.handle, nb::cast<nb::dict>(item), recursive, visited);
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
                        auto& child = root.AddChild(node, fmt::format("{}.[list]#{}.[list]#{}", key, i, j));
                        child.data  = StateNode::Dict{};
                        if (!nb::isinstance<nb::dict>(item1))
                        {
                            throw std::runtime_error(
                                fmt::format("[TODO]StateDict::Register] Unsupported list element type for {}. Expected dict, got {}",
                                            child.name,
                                            nb::str(item1.type())));
                        }
                        // nanobind::gil_scoped_release release;
                        RegisterStateDictRecr(root, child.handle, nb::cast<nb::dict>(item1), recursive, visited);
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
    else if (TryRegisterTorchStateDict(root, node, container, recursive, visited, key, val)) {}
    else if (recursive)
    {
        if (nb::isinstance<nb::dict>(val))
        {
            // nanobind::gil_scoped_release release;
            auto& child = root.AddChild(node, keyView);
            child.data  = StateNode::Dict{};
            RegisterStateDictRecr(root, child.handle, nb::cast<nb::dict>(val), recursive, visited);
        }
        // if namespace
        else if (IsNamespace(val))
        {
            // nanobind::gil_scoped_release release;
            auto& child = root.AddChild(node, keyView);
            child.data  = StateNode::Dict{};
            RegisterStateDictRecr(root, child.handle, nb::cast<nb::dict>(val.attr("__dict__")), recursive, visited);
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

void RegisterStateDictRecr(StateTree&                     root,
                           StateNode::Handle              node,
                           nb::dict const&                container,
                           bool                           recursive,
                           std::unordered_set<PyObject*>& visited)
{
    if (visited.contains(container.ptr())) { return; }
    visited.insert(container.ptr());

    for (auto const& item : container)
    {
        nanobind::gil_scoped_acquire acquire;
        try
        {
            RegisterStateDictRecrContainerItem(root, node, container, recursive, visited, nb::str(item.first), item.second);
        } catch (std::exception const& e)
        {
            fmt::print(stderr,
                       "[native_ckpt:cli][ERROR] Error processing prefix = {}, key: {}: {}\n",
                       root.GetNodeId(node),
                       nb::str(item.first),
                       e.what());
            throw;
        }
    }
}

static void RegisterStateDict(StateTree& self, nb::dict const& container)
{
    if (self.GetNode(StateNode::Handle::Root).handler == nullptr)
    {
        std::unordered_set<PyObject*> visited;
        RegisterStateDictRecr(self, StateNode::Handle::Root, container, true, visited);
        self.GetNode(StateNode::Handle::Root).handler = std::make_unique<RegisteredStateDict>(container);
        return;
    }

    auto* registered = dynamic_cast<RegisteredStateDict*>(self.GetNode(StateNode::Handle::Root).handler.get());
    if (registered == nullptr) { throw std::logic_error("state tree root already has a non-dictionary handler"); }

    registered->Merge(container);
    self.WalkNodes([&](StateNode& node) {
        if (node.handle != StateNode::Handle::Root && node.handler != nullptr) { self.RefreshNodeDataFromHandler(node); }
    });
}

static void FinalizeRegistry(StateTree& self)
{ FinalizeStateRegistry(self); }

static void WaitForDeferredSave(StateTree& self)
{
    nanobind::gil_scoped_release release;
    self.WaitForAsyncCalls();
}

static void SaveAsync(StateTree& self, int iteration, std::filesystem::path const& dirpath)
{
    auto                         snapshot = self.Clone();
    nanobind::gil_scoped_release release;

    self.ChainAsyncCall([iteration, dirpath, snapshot = std::move(snapshot)]() {
        SaveCheckpoint(*snapshot, static_cast<cipc::SnapshotId>(iteration), dirpath);
    });
}

static void Save(StateTree& self, int iteration, std::filesystem::path const& dirpath)
{
    auto                         snapshot = self.Clone();
    nanobind::gil_scoped_release release;
    SaveCheckpoint(*snapshot, static_cast<cipc::SnapshotId>(iteration), dirpath);
}

NB_MODULE(checkpoint, m)    // NOLINT
{
    m.def("ping", &SvcPing);
    m.def("svc_stop", &SvcStop);
    m.def("svc_spindown", &SvcSpinDown /* , nanobind::arg("timeout") = nanobind::none() */);
    m.def(
        "wait_for_ckpt_finalize",
        [](uint32_t sessionId, int64_t iteration) {
            nb::gil_scoped_release release;
            WaitForCkptFinalize(sessionId, iteration);
        },
        nb::arg("session_id"),
        nb::arg("iteration"));
    auto cls = nb::class_<StateTree>(std::move(m), "model_state");

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
               return std::make_unique<StateTree>(RankTopology{
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
        .def_static(
            "warmup",
            [](uint32_t                 globalRank,
               uint32_t                 globalWorldSize,
               std::vector<std::string> peers,
               uint32_t                 ranksPerPeer,
               uint32_t                 peersPerNode) {
                WarmUpCheckpoint(RankTopology{
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
            nb::arg("ranks_per_peer"),
            nb::arg("peers_per_node") = 1)
        .def("add_child", &StateTree::AddChild, nb::rv_policy::reference)
        .def_prop_ro("globalRank", &StateTree::GlobalRank)
        .def_prop_ro("globalWorldSize", &StateTree::GlobalWorldSize)
        .def_prop_rw(
            "startedAt",
            [](StateTree const& self) { return self.startedAt.count(); },
            [](StateTree& self, int64_t startedAt) { self.startedAt = std::chrono::nanoseconds{startedAt}; })
        .def("register_state_dict", &RegisterStateDict)
        .def("finalize_registry", &FinalizeRegistry)
        .def("save_async", &SaveAsync, nb::arg("iteration"), nb::arg("dirpath"))
        .def("save", &Save, nb::arg("iteration"), nb::arg("dirpath"))
        .def("wait_for_deferred_save", &WaitForDeferredSave)

        // Save/Load convenience methods that block until complete
        //.def("save",
        //     [](std::shared_ptr<StateTree> const& self, std::string const& path) {
        //     self->SaveAsync(std::filesystem::path(path)).get(); })
        //.def("load",
        //     [](std::shared_ptr<StateTree> const& self, std::string const& path) {
        //     self->LoadAsync(std::filesystem::path(path)).get(); })

        // Return list of named states' names (as Python list of strings)
        ;
}
