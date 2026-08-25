
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

static void Register(StateTree& self, StateNode::Handle node, nb::str const& name, nb::object& value)
{
    // check if immutable type
    if (nb::isinstance<int64_t>(value) || nb::isinstance<double>(value) || nb::isinstance<bool>(value) || nb::isinstance<nb::str>(value))
    {
        throw std::runtime_error(
            fmt::format("Immutable types are not supported. Please pass locals() or a dictionary containing the value"));
    }

    // if dictionary
    if (nb::isinstance<nb::dict>(value))
    {
        auto d = nb::cast<nb::dict>(value);
#ifndef NDEBUG
        if (!d.contains(name)) { throw std::runtime_error(fmt::format("Dictionary does not contain key: {}", name)); }
#endif
        auto obj = d.attr(name);
        if (obj.is_none()) { throw std::runtime_error(fmt::format("Dictionary value is None for key: {}", name)); }
        if (nb::isinstance<int64_t>(obj) || nb::isinstance<double>(obj) || nb::isinstance<bool>(obj))
        {
            self.RegisterStateHandler(node, std::make_unique<StateTreeScalarHandler>(std::move(d), name));
            return;
        }
        if (nb::isinstance<nb::str>(obj))
        {
            self.RegisterStateHandler(node, std::make_unique<StateTreeStringHandler>(std::move(d), name));
            return;
        }
        throw std::runtime_error(fmt::format("Unsupported type for key: {}. Supported types are int, float, bool, str", name));
    }

    throw std::runtime_error(fmt::format("register_scalar: {}, {}", name, nb::str(value.type())));
}

static bool IsEnum(nb::handle const& obj)
{
    static auto Enum = nb::module_::import_("enum").attr("Enum");
    return nb::isinstance(obj, Enum);
}

static bool IsSimpleNamespace(nb::handle obj)
{
    nb::object types         = nb::module_::import_("types");
    nb::object namespaceType = types.attr("SimpleNamespace");
    return nb::isinstance(obj, namespaceType);
}

static bool IsArgParseNamespace(nb::handle obj)
{
    nb::object argparse      = nb::module_::import_("argparse");
    nb::object namespaceType = argparse.attr("Namespace");
    return nb::isinstance(obj, namespaceType);
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
    auto                          start = std::chrono::system_clock::now();
    std::unordered_set<PyObject*> visited;
    RegisterStateDictRecr(self, StateNode::Handle::Root, container, true, visited);
    auto stop = std::chrono::system_clock::now();
}

static void RegisterStateDictAsync(StateTree& self, nb::dict const& container)
{
    self.ChainAsyncCall([&self, container]() { RegisterStateDict(self, container); });
}

static void RegisterScalarsInDict(StateTree& self, nb::dict& container)
{
    std::unordered_set<PyObject*> visited;
    RegisterStateDictRecr(self, StateNode::Handle::Root, container, false, visited);
}

static void RegisterDeferredSaveTensor(StateTree& self, std::string_view const& name, nb::object& value)
{}

static void RegisterImmediateSaveTensor(StateTree& self, std::string_view const& name, nb::object& value)
{}

static void FinalizeRegistry(StateTree& self)
{ FinalizeStateRegistry(self); }

static void WaitForDeferredSave(StateTree& self)
{
    nanobind::gil_scoped_release release;
    self.WaitForAsyncCalls();
}

static void SaveAsync(StateTree& self, int iteration, std::filesystem::path const& dirpath)
{
    auto                         snapshotBlobs = self.CaptureSnapshotBlobs();
    nanobind::gil_scoped_release release;

    self.ChainAsyncCall([&self, iteration, dirpath, snapshotBlobs = std::move(snapshotBlobs)]() {
        // self.Update();
        SaveCheckpoint(self, static_cast<cipc::SnapshotId>(iteration), dirpath, snapshotBlobs);
    });
}

static void Save(StateTree& self, int iteration, std::filesystem::path const& dirpath)
{
    auto                         snapshotBlobs = self.CaptureSnapshotBlobs();
    nanobind::gil_scoped_release release;
    // self.Update();
    SaveCheckpoint(self, static_cast<cipc::SnapshotId>(iteration), dirpath, snapshotBlobs);
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
        .def("register", &Register)
        .def("register_state_dict", &RegisterStateDict)
        .def("refresh_sharded_tensors", &RefreshShardedTensors, nb::arg("fresh_list"))
        .def("register_state_dict_async", &RegisterStateDictAsync)
        .def("register_scalars_in_dict", &RegisterScalarsInDict)
        .def("register_weight_tensor", &RegisterDeferredSaveTensor)
        .def("register_immediate_tensor", &RegisterImmediateSaveTensor)
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
