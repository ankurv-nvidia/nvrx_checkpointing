
#include "binding.h"
#include "state.h"
#include "tensor.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <torch/csrc/Dtype.h>
#include <torch/serialize.h>
#include <torch/torch.h>

#include <algorithm>

#include <ranges>

#define TODO_SHTEN_KEY
namespace
{

struct THPVariable
{
    PyObject_HEAD
#if defined TORCH_ABI_VERSION && TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 10
        at::Tensor cdata;
#else
        // Payload
            c10::MaybeOwned<at::Tensor>
            cdata;
#endif
    // Hooks to be run on backwards pass (corresponds to Python attr
    // '_backwards_hooks', set by 'register_hook')
    PyObject* backward_hooks = nullptr;    // NOLINT
    // Hooks to be run in the backwards pass after accumulate grad,
    // i.e., after the .grad has been set (corresponds to Python attr
    // '_post_accumulate_grad_hooks', set by 'register_post_accumulate_grad_hook')
    PyObject* post_accumulate_grad_hooks = nullptr;    // NOLINT
};
}    // namespace

namespace nb = nanobind;
static void LoadDcpSafeTensors(StateTree& self, std::filesystem::path const& path)
{
    torch::serialize::InputArchive archive;
    archive.load_from((path / "__0_0.distcp").string());

    torch::Tensor t;
    archive.read("weight", t);    // key depends on what was saved
}

static constexpr DS4Tensor::DType TorchDtypeMap(at::ScalarType dtype)
{
    switch (dtype)
    {
    case at::ScalarType::BFloat16: return DS4Tensor::DType::BFloat16;
    case at::ScalarType::Half: return DS4Tensor::DType::Float16;
    case at::ScalarType::Float: return DS4Tensor::DType::Float32;
    case at::ScalarType::Double: return DS4Tensor::DType::Float64;

    case at::ScalarType::Long: return DS4Tensor::DType::Int64;    // NOLINT
    case at::ScalarType::Int: return DS4Tensor::DType::Int64;
    case at::ScalarType::Short: return DS4Tensor::DType::Int16;
    case at::ScalarType::Char: return DS4Tensor::DType::Int8;

    case at::ScalarType::Byte: return DS4Tensor::DType::UInt8;
    default: throw std::runtime_error(fmt::format("Unsupported torch dtype: {}", static_cast<int>(dtype)));
    }
}

TORCH_PYTHON_API extern PyTypeObject* THPVariableClass;    // NOLINT
TORCH_PYTHON_API extern PyTypeObject* ParameterClass;      // NOLINT

static bool THPVariableCheckTypeExact(PyTypeObject* tp)
{
    // Check that a python object is a `Tensor`, but not a `Tensor` subclass.
    // (A subclass could have different semantics.) The one exception is
    // Parameter, which is used for Python bookkeeping but is equivalent to
    // Tensor as far as C++ is concerned.
    return (tp == THPVariableClass) || (tp == ParameterClass);
}

static bool THPVariableCheckExact(PyObject* obj)
{ return THPVariableCheckTypeExact(Py_TYPE(obj)); }

static bool THPVariableCheck(PyObject* obj)
{
    if (THPVariableClass == nullptr) { return false; }

    // Fast path
    if (THPVariableCheckExact(obj)) { return true; }

    auto const result = PyObject_IsInstance(obj, reinterpret_cast<PyObject*>(THPVariableClass));    // NOLINT
    if (result == -1) { throw nb::python_error(); }
    return result != 0;
}

static at::Tensor const& ThpVariableUnpack(THPVariable* var)
{
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 10
    return var->cdata;
#else
    return *var->cdata;
#endif
}

static at::Tensor const& ThpVariableUnpack(PyObject* obj)
{
    return ThpVariableUnpack(reinterpret_cast<THPVariable*>(obj));    // NOLINT
}

static bool IsTorchDtype(nb::handle obj)
{ return THPDtype_Check(obj.ptr()); }

static bool IsMegatronShardedTensor(nb::handle obj)
{
    nb::object megatron          = nb::module_::import_("megatron.native.mapping");
    nb::object shardedTensorType = megatron.attr("ShardedTensor");
    return nb::isinstance(obj, shardedTensorType);
}
static bool IsMegatronShardedTensorFactory(nb::handle obj)
{
    nb::object megatron          = nb::module_::import_("megatron.native.mapping");
    nb::object shardedTensorType = megatron.attr("ShardedTensorFactory");
    return nb::isinstance(obj, shardedTensorType);
}

static bool IsMegatronShardedObject(nb::handle obj)
{
    nb::object megatron          = nb::module_::import_("megatron.native.mapping");
    nb::object shardedObjectType = megatron.attr("ShardedObject");
    return nb::isinstance(obj, shardedObjectType);
}

namespace
{
struct StateTreeTorchDTypeHandler : StateTree::StateHandler<StateTree::StateType::Scalar>
{
    StateTreeTorchDTypeHandler(nb::dict containerIn, nb::str const& nameIn) :
        container(std::move(containerIn)), StateTree::StateHandler<StateTree::StateType::Scalar>(StrView(nameIn)), key(nameIn)
    {}

    void Update() override {};

    StateTree::Scalar GetScalarState() override
    {
        nb::gil_scoped_acquire acquire;

        auto obj   = container[key];
        auto dtype = reinterpret_cast<THPDtype*>(obj.ptr());    // NOLINT
        return {static_cast<int64_t>(dtype->scalar_type)};
    }

    void SetScalarState(StateTree::Scalar const& value) override
    {
        nb::gil_scoped_acquire acquire;

        auto obj           = container[key];
        auto dtype         = reinterpret_cast<THPDtype*>(obj.ptr());    // NOLINT
        dtype->scalar_type = static_cast<at::ScalarType>(value.Cast<int64_t>());
    };

    nb::dict container;
    nb::str  key;
};

struct StateTreeTorchTensorHandler : StateTree::StateHandler<StateTree::StateType::TensorShard>
{
    StateTreeTorchTensorHandler(nb::dict containerIn, nb::str const& nameIn) :
        container(std::move(containerIn)), StateTree::StateHandler<StateTree::StateType::TensorShard>(StrView(nameIn)), key(nameIn)
    { Update(); }

    void Update() override
    {
        // nb::gil_scoped_acquire acquire;
        // meta      = {};
        auto tnsr = ThpVariableUnpack(container[key].ptr());
        if (!tnsr.is_contiguous()) { throw std::runtime_error(fmt::format("Tensor {} is not contiguous", name)); }
        auto device = tnsr.device();
        meta.device = DeviceId(device.has_index() ? device.str() : fmt::format("{}:{}", device.str(), 0));
        meta.dtype  = TorchDtypeMap(tnsr.dtype().toScalarType());
        meta.handle = reinterpret_cast<uint64_t>(tnsr.data_ptr());    // NOLINT
        auto nbytes = tnsr.nbytes();
        if (nbytes == 0) { return; }

        if (tnsr.sizes().size() > StateNode::TensorShard::MaxDims)
        {
            throw std::runtime_error(fmt::format("Tensor {} has too many dimensions: {}", name, tnsr.sizes().size()));
        }
        if (tnsr.sizes().size() == 0)
        {
            meta.fulldim.push_back(1);
            meta.sharddim.push_back(DS4Tensor::Range{.begin = 0, .end = 1});
        }
        else
        {
            if (tnsr.size(0) <= 0)
            {
                throw std::runtime_error(fmt::format("Tensor {} has zero dimension sizes : [{}]", name, fmt::join(tnsr.sizes(), ", ")));
            }

            for (auto dim : tnsr.sizes())
            {
                if (dim == 0)
                {
                    throw std::runtime_error(fmt::format("Tensor {} has zero dimension sizes: [{}]", name, fmt::join(tnsr.sizes(), ", ")));
                }
                meta.fulldim.push_back(dim);
                meta.sharddim.push_back(DS4Tensor::Range{.begin = 0, .end = static_cast<DS4Tensor::SizeT>(dim)});
            }
        }
        if (meta.ShardSizeBytes() != tnsr.nbytes()) { throw std::runtime_error(fmt::format("Tensor {} size mismatch", name)); }
        meta.AssertShardIsValid();
    }

    DS4Tensor GetTensorState() override { return meta; }

    void SetTensorState(DS4Tensor const& value) override
    {
        nb::gil_scoped_acquire acquire;
        if (GetTensorState() != value) { throw std::runtime_error("SetTensorState mismatch"); }
        auto tnsr = ThpVariableUnpack(container[key].ptr());
    };

    DS4Tensor meta{};
    nb::dict  container;
    nb::str   key;
};

struct StateTreeTorchGeneratorHandler : StateTree::StateHandler<StateTree::StateType::Blob>
{
    StateTreeTorchGeneratorHandler(nb::dict containerIn, nb::str const& nameIn) :
        container(std::move(containerIn)), StateTree::StateHandler<StateTree::StateType::Blob>(StrView(nameIn)), key(nameIn)
    {}

    void Update() override {}

    std::vector<char> GetBlob() override
    {
        nb::gil_scoped_acquire acquire;
        auto                   state = container[key].attr("get_state")();

        if (!THPVariableCheck(state.ptr()))
        {
            throw std::runtime_error(fmt::format("Expected torch.Generator::get_state to return a tensor for key: {}", name));
        }
        auto tnsr               = ThpVariableUnpack(state.ptr());
        tnsr                    = tnsr.to(at::DeviceType::CPU);
        auto              bytes = tnsr.nbytes();
        std::vector<char> data(bytes);
        auto              it = std::copy(
            reinterpret_cast<char*>(tnsr.data_ptr()), reinterpret_cast<char*>(tnsr.data_ptr()) + bytes, data.begin());    // NOLINT
        assert(it == data.end());
        return data;
    }

    void SetBlob(std::string_view const& value) override
    {
        nb::gil_scoped_acquire acquire;
        throw std::runtime_error("TODO");
        // if (nb::isinstance<nb::str>(container)) { container[key] = std::string(value); }
        // else
        //{
        //     throw std::out_of_range("Conversion of Blob to nb::object failed");
        // }
    }

    nb::dict container;
    nb::str  key;
};

struct StateTreeShardedTensorHandler : StateTree::StateHandler<StateTree::StateType::TensorShard>
{
    public:
    static std::unique_ptr<StateTreeShardedTensorHandler> Create(nb::dict const& containerIn, nb::str nameIn)
    {
        auto meta = Convert(nameIn, containerIn);
        return std::unique_ptr<StateTreeShardedTensorHandler>(new StateTreeShardedTensorHandler(containerIn, nameIn, meta));
    }

    private:
    StateTreeShardedTensorHandler(nb::dict containerIn, nb::str const& nameIn, DS4Tensor const& metaIn) :
        StateTree::StateHandler<StateTree::StateType::TensorShard>(StrView(nameIn)),
        meta(metaIn),
        container(std::move(containerIn)),
        key(nameIn)
    {}

    void Update() override {}

    static DS4Tensor Convert(nb::str& name, nb::dict const& container)    // NOLINT
    {
        nb::gil_scoped_acquire acquire;

        using Range = DS4Tensor::Range;
        DS4Tensor ds4tensor{};
        auto      shtKey     = nb::cast<nb::str>(container["key"]);
        auto      nameView   = StrView(name);
        auto      shtKeyView = StrView(shtKey);
        if (!nameView.starts_with(shtKeyView))
        {
#if !defined TODO_SHTEN_KEY
            fmt::print("[native_ckpt:cli][WARN] ShardedTensor state-dict key does not match(prefix) tensor key. Using sharded tensor "
                       "key.This can cause unexpected behavior.\n"
                       "\tstate-dict     key = {}\n"
                       "\tsharded-tensor key = {}\n",
                       nameView,
                       shtKeyView);
#endif
            name     = shtKey;
            nameView = shtKeyView;
        }

        auto tnsr           = ThpVariableUnpack(nb::cast<nb::object>(container["data"]).ptr());
        auto globalShape    = nb::cast<nb::tuple>(container["global_shape"]);
        auto localShape     = nb::cast<nb::tuple>(container["local_shape"]);
        auto globalOffset   = nb::cast<nb::tuple>(container["global_offset"]);
        bool printError     = false;
        auto replicaId      = container["replica_id"];
        auto flattenedRange = container["flattened_range"];

        if (!tnsr.is_contiguous()) { throw std::runtime_error(fmt::format("Tensor {} is not contiguous", shtKeyView)); }
        ds4tensor.dtype  = TorchDtypeMap(tnsr.dtype().toScalarType());
        auto device      = tnsr.device();
        ds4tensor.device = DeviceId(device.has_index() ? device.str() : fmt::format("{}:{}", device.str(), 0));
        struct ShardedTensorError : public std::runtime_error
        {
            ShardedTensorError(nb::str const& key, nb::dict const& container, std::string_view const& message) :
                std::runtime_error(fmt::format("Error Registering StateDict ShardedTensor::{} key = {} shardedtensor = {}",
                                               message,
                                               key,
                                               nb::str(container)))
            {}
        };
        auto throwError = [&](std::string_view const& message) { throw ShardedTensorError(shtKey, container, message); };

        if (globalShape.size() != globalOffset.size())
        {
            throwError(fmt::format(
                "global_shape size ([{}]) mismatch with global_offset size ([{}])", nb::str(globalShape), nb::str(globalOffset)));
        }

        [&]() {
            auto prependAxisNum = container["prepend_axis_num"];
            if (prependAxisNum.is_none() && globalShape.size() == localShape.size()) { return; }
            if (prependAxisNum.is_none() && globalShape.size() != localShape.size())
            {
                throwError(fmt::format("prepend_axis_num is None inconsistent with global_shape([{}]), local_shape([{}])",
                                       nb::str(globalShape),
                                       nb::str(localShape)));
            }
            if (!nb::isinstance<int64_t>(prependAxisNum)) { throwError("prependAxisNum is not int"); }
            auto prependAxisNumValue = nb::cast<int64_t>(prependAxisNum);
            if (prependAxisNumValue == 0 && globalShape.size() == localShape.size()) { return; }
            if (prependAxisNumValue == 1 && globalShape.size() == (localShape.size() + 1)) { return; }
            throwError(fmt::format("prepend_axis_num = {} inconsistent with lobal_shape([{}]), local_shape([{}])\n",
                                   shtKeyView,
                                   prependAxisNumValue,
                                   nb::str(globalShape),
                                   nb::str(localShape)));
        }();

        [&]() {
            auto allowShapeMismatch = container["allow_shape_mismatch"];
            if (allowShapeMismatch.is_none()) { return; }
            if (!nb::isinstance<bool>(allowShapeMismatch)) { throwError("allow_shape_mismatch is not bool"); }
            if (!nb::cast<bool>(allowShapeMismatch)) { return; }
            fmt::print(stderr,
                       "[native_ckpt:cli][TODO][StateDict::Register::ShardedTensor] key = {}, allow_shape_mismatch = {} globalShape = [{}] "
                       "localShape=[{}] "
                       "globalOffset=[{}]\n",
                       shtKeyView,
                       true,
                       nb::str(globalShape),
                       nb::str(localShape),
                       nb::str(globalOffset));
        }();

        if (!flattenedRange.is_none()) { throwError("flattened_range is not None"); }

        [&]() {
            auto axisFragmentations = container["axis_fragmentations"];
            if (axisFragmentations.is_none()) { return; }
            if (!nb::isinstance<nb::tuple>(axisFragmentations)) { throwError("axis_fragmentations is not tuple"); }
            auto axisFragmentationsTuple = nb::cast<nb::tuple>(axisFragmentations);
            if (axisFragmentationsTuple.size() != globalShape.size())
            {
                throwError(fmt::format("axis_fragmentations size ([{}]) mismatch with global_shape size ([{}])",
                                       nb::str(axisFragmentationsTuple),
                                       nb::str(globalShape)));
            }
            size_t localDim = 0;

            for (size_t i = 0; i < axisFragmentationsTuple.size(); ++i)
            {
                if (!nb::isinstance<int64_t>(axisFragmentationsTuple[i]))
                {
                    throwError(fmt::format("axis_fragmentations is not tuple[int] at index {}", i));
                }

                auto fragVal      = nb::cast<int64_t>(axisFragmentationsTuple[i]);
                auto globalDimVal = nb::cast<int64_t>(globalShape[i]);

                if (globalShape.size() != localShape.size() && i == 0)
                {
                    if (fragVal != globalDimVal)
                    {
                        throwError("axis_fragmentations[0] does not match global_shape[0] for prepend_axis_num=1 case");
                    }
                    continue;
                }
                auto localDimVal = nb::cast<int64_t>(localShape[localDim++]);
                if (globalDimVal != fragVal * localDimVal)
                {
                    throwError(fmt::format("axis_fragmentations[{}] != global_shape[{}] / local_shape[{}] ({} != {} / {})",
                                           i,
                                           i,
                                           i,
                                           fragVal,
                                           globalDimVal,
                                           localDimVal));
                }
            }
        }();

        [&]() {
            if (replicaId.is_none()) { return; }
            if (nb::isinstance<nb::tuple>(replicaId))
            {
                for (auto const& val : nb::cast<nb::tuple>(replicaId))
                {
                    if (!nb::isinstance<int64_t>(val)) { throwError("replicaId is not tuple[int]"); }
                    if (nb::cast<int64_t>(val) != 0)
                    {
#if defined TODO_REPLICA_ID_TUPLE
                        fmt::print(stderr,
                                   "[native_ckpt:cli][TODO][StateDict::Register::ShardedTensor::ReplicaId] key = {}, replica_id = {} \n",
                                   shtKeyView,
                                   nb::str(nb::cast<nb::tuple>(replicaId)));
#endif
                        return;
                    }
                }
                return;
            }
            if (nb::isinstance<int64_t>(replicaId))
            {
                if (nb::cast<int64_t>(replicaId) != 0)
                {
                    fmt::print(stderr,
                               "[native_ckpt:cli][TODO][StateDict::Register::ShardedTensor::ReplicaId] key = {}, replica_id = {} \n",
                               shtKeyView,
                               nb::str(nb::cast<nb::tuple>(replicaId)));
                    return;
                }
                return;
            }
            throwError("replicaId is not tuple");
        }();

        /*axis_fragmentations: Optional[Tuple[int, ...]]
        replica_id:  Union[int, Tuple[int, ...]] = 0
        prepend_axis_num: int = 0
        allow_shape_mismatch: bool = False
        flattened_range: Optional[slice] = None*/

        for (auto const& val : globalShape) { ds4tensor.fulldim.push_back(nb::cast<int64_t>(val)); }
        for (size_t i = 0; i < globalShape.size(); ++i)
        {
            auto    start               = nb::cast<int64_t>(globalOffset[i]);
            auto    prependAxisNumValue = nb::cast<int64_t>(container["prepend_axis_num"]);
            int64_t end{};
            if (i < prependAxisNumValue) { end = start + 1; }
            else
            {
                end = start + nb::cast<int64_t>(localShape[i - prependAxisNumValue]);
            }
            ds4tensor.sharddim.push_back(Range{.begin = static_cast<size_t>(start), .end = static_cast<size_t>(end)});
        }

        if (ds4tensor.ShardSizeBytes() != tnsr.nbytes()) { throw std::runtime_error(fmt::format("Tensor {} size mismatch", shtKeyView)); }
        ds4tensor.handle = reinterpret_cast<uint64_t>(tnsr.data_ptr());    // NOLINT
        ds4tensor.AssertShardIsValid();
        return ds4tensor;
    }

    public:
    DS4Tensor GetTensorState() override { return meta; }

    void SetTensorState(DS4Tensor const& value) override
    {
        nb::gil_scoped_acquire acquire;
        throw std::runtime_error("SetTensorState not implemented for ShardedTensor");
    };

    DS4Tensor meta;
    nb::dict  container;
    nb::str   key;
};
}    // namespace

static void RegisterShardedTorchTensorFactory(StateTree& self, StateNode::Handle node, nb::str const& name, nb::object const& container)
{
    auto factoryKey     = nb::cast<nb::str>(container.attr("key"));
    auto factoryKeyView = StrView(factoryKey);
    auto nameView       = StrView(name);
    if (factoryKeyView != nameView)
    {
        throw std::runtime_error(fmt::format("ShardedTensorFactory key mismatch: expected {}, got {}", nameView, factoryKeyView));
    }
    // Call the build() function on the container and get the result which should be a list of shardedtensors prefixed with key
    auto buildFunc      = container.attr("build");
    auto shardedTensors = buildFunc();
    if (!nb::isinstance<nb::list>(shardedTensors))
    {
        throw std::runtime_error(fmt::format("Expected ShardedTensorFactory::build to return a tuple of ShardedTensors for key: {}", name));
    }

    auto shardedTensorsTuple = nb::cast<nb::list>(shardedTensors);
    for (auto const& item : shardedTensorsTuple)
    {
        if (!IsMegatronShardedTensor(item))
        {
            throw std::runtime_error(
                fmt::format("Expected ShardedTensorFactory::build to return a tuple of ShardedTensors for key: {} but got {}",
                            name,
                            nb::str(item.type())));
        }
        auto container  = nb::cast<nb::dict>(item.attr("__dict__"));
        auto shtKey     = nb::cast<nb::str>(container["key"]);
        auto shtKeyView = StrView(shtKey);
        auto itdot      = std::ranges::find(std::views::reverse(shtKeyView), '.');
        if (itdot != shtKeyView.rend())
        {
            auto fqn = nameView.contains('.') ? std::string(nameView) : fmt::format("{}.{}", self.GetNodeId(node), nameView);
            if (!shtKeyView.starts_with(fqn) || shtKeyView[fqn.size()] != '.')
            {
                throw std::runtime_error(fmt::format(
                    "ShardedTensorFactory::build returned ShardedTensor with key {} that is not a child of the factory obj fqn {}",
                    shtKeyView,
                    fqn));
            }
        }
        self.RegisterStateHandler(node, StateTreeShardedTensorHandler::Create(container, shtKey));
    }
}

static void RegisterShardedObject(StateTree& self, StateNode::Handle node, std::string_view const& name, nb::dict const& container)
{
    [[maybe_unused]] auto tensordata   = container.attr("data");
    [[maybe_unused]] auto key          = container.attr("key");
    [[maybe_unused]] auto globalshape  = container.attr("global_shape");
    [[maybe_unused]] auto globalOffset = container.attr("global_offset");
}

static bool IsTorchGenerator(nb::handle obj)
{
    nb::object torch         = nb::module_::import_("torch");
    nb::object generatorType = torch.attr("Generator");
    return nb::isinstance(obj, generatorType);
}

// After `apply_factories`, a key like `decoder.layers.0.mixer.in_proj.weight`
// gets replaced with `[ShardedTensor(...weight.z), ShardedTensor(...weight.x),
// ShardedTensor(...weight.B), ShardedTensor(...weight.C), ShardedTensor(...weight.dt)]`.
// Register each element using its own `.key` as the FQN — same shape as nvrx's
// post-`save_preprocess` state_dict, so the cross-provider tensor compare can
// match tensors per-FQN. Returns false when the list isn't an all-ShardedTensor
// list so the generic walker can fall through to its existing per-element-type
// handling.
bool TryRegisterTorchStateDictList(StateTree& root, StateNode::Handle node, nb::str const& key, nb::list const& listObj)
{
    if (listObj.size() == 0) { return false; }
    for (auto const& item : listObj)
    {
        if (!IsMegatronShardedTensor(item)) { return false; }
    }
    auto const nameView = StrView(key);
    // Megatron's flattened-state-dict convention: when the state-dict key
    // already has dots (e.g. `decoder.layers.0.mixer.in_proj.weight`) it IS
    // the FQN — Megatron stashes such pre-flattened keys inside the top-level
    // `model` / `optimizer` / etc dicts. Treat such names as already-qualified
    // and skip the parent-node prefix; otherwise compose the FQN from the
    // parent node id (mirrors `RegisterShardedTorchTensorFactory`).
    auto const fqn = nameView.contains('.') ? std::string(nameView) : fmt::format("{}.{}", root.GetNodeId(node), nameView);
    for (auto const& item : listObj)
    {
        auto itemContainer = nb::cast<nb::dict>(item.attr("__dict__"));
        auto shtKey        = nb::cast<nb::str>(itemContainer["key"]);
        auto shtKeyView    = StrView(shtKey);
        // Each ShardedTensor's `.key` is the canonical FQN. It must equal
        // the surrounding state-dict FQN (rare: a single-element list) or be
        // a dotted child of it. Anything else is a structural surprise we'd
        // rather raise on.
        if (shtKeyView != fqn && (!shtKeyView.starts_with(fqn) || shtKeyView.size() <= fqn.size() || shtKeyView[fqn.size()] != '.'))
        {
            throw std::runtime_error(
                fmt::format("ShardedTensor list element key '{}' is not a child of state-dict key '{}'", shtKeyView, fqn));
        }
        root.RegisterStateHandler(node, StateTreeShardedTensorHandler::Create(itemContainer, shtKey));
    }
    return true;
}

bool TryRegisterTorchStateDict(StateTree&                     root,
                               StateNode::Handle              node,
                               nb::dict const&                container,
                               bool                           recursive,
                               std::unordered_set<PyObject*>& visited,
                               nb::str const&                 key,
                               nb::handle                     val)
{
    if (THPVariableCheck(val.ptr()))
    {
        auto tnsr = ThpVariableUnpack(val.ptr());
        root.RegisterStateHandler(node, std::make_unique<StateTreeTorchTensorHandler>(container, key));
        return true;
    }
    if (IsTorchDtype(val))
    {
        root.RegisterStateHandler(node, std::make_unique<StateTreeTorchDTypeHandler>(container, key));
        return true;
    }
    if (IsMegatronShardedTensor(val))
    {
        auto container = nb::cast<nb::dict>(val.attr("__dict__"));
        auto tnsr      = ThpVariableUnpack(nb::cast<nb::object>(container["data"]).ptr());
        root.RegisterStateHandler(node, StateTreeShardedTensorHandler::Create(container, key));
        return true;
    }
    if (IsMegatronShardedTensorFactory(val))
    {
        RegisterShardedTorchTensorFactory(root, node, key, nb::borrow<nb::object>(val));
        return true;
    }
    if (IsMegatronShardedObject(val))
    {
#if defined TODO_SHRDOBJ
        fmt::print(stderr,
                   "[native_ckpt:cli][{}][TODO] RegisterShardedObject for parent = {} key = {}, obj = {}\n",
                   root.GlobalRank(),
                   root.GetNodeId(node),
                   key,
                   nb::str(val));
#endif
        // TODO_SHRDOBJ
        // auto& child = root.AddChild(node, fmt::format("{}.#sharded_object#", key));
        // child.data  = StateNode::Dict{};
        // RegisterStateDictRecr(root, child.handle, nb::cast<nb::dict>(val.attr("__dict__")), recursive, visited);
        return true;
    }
    if (IsTorchGenerator(val))
    {
        root.RegisterStateHandler(node, std::make_unique<StateTreeTorchGeneratorHandler>(container, key));
        return true;
    }
    return false;
}

void AddTorchBindings(nb::class_<StateTree>& cls)
{
    cls.def("load_dcp_safetensors", &LoadDcpSafeTensors);
    // cls.def("register_torch_tensor", &RegisterTorchTensor)
    //     .def("register_sharded_torch_tensor", &RegisterShardedTorchTensor)
    //     .def("register_sharded_torch_tensor_factory", &RegisterShardedTorchTensorFactory);
}