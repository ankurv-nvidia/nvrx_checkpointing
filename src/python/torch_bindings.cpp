#include "binding.h"
#include "state.h"
#include "tensor.h"

#include <nanobind/nanobind.h>

#include <dlpack/dlpack.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>

namespace nb = nanobind;

namespace
{

struct DtypeIdEntry
{
    std::string_view name;
    int64_t          id;
};
// Holds the capsule alongside a pointer into its DLTensor. The capsule
// must outlive any read of the DLTensor fields it points to.
struct DlpackCapsule
{
    nb::object      capsule;
    DLTensor const* tnsr;
};

constexpr auto KDtypeIds = std::array<DtypeIdEntry, 13>{
    {
        {.name = "torch.float16", .id = 1},
        {.name = "torch.bfloat16", .id = 2},
        {.name = "torch.float32", .id = 3},
        {.name = "torch.float64", .id = 4},
        {.name = "torch.uint8", .id = 5},
        {.name = "torch.uint16", .id = 6},
        {.name = "torch.uint32", .id = 7},
        {.name = "torch.uint64", .id = 8},
        {.name = "torch.int8", .id = 9},
        {.name = "torch.int16", .id = 10},
        {.name = "torch.int32", .id = 11},
        {.name = "torch.int64", .id = 12},
        {.name = "torch.bool", .id = 13},
    },
};

struct CachedTypes
{
    nb::object torchTensor;
    nb::object torchDtype;
    nb::object torchGenerator;
    nb::object localShardedTensor;
    nb::object localShardedTensorFactory;
    nb::object localShardedObject;
    nb::object megatronShardedTensor;
    nb::object megatronShardedTensorFactory;
    nb::object megatronShardedObject;
};

struct TensorState
{
    DTensor    meta;
    nb::object owner;
};
}    // namespace

static bool IsMainReplica(nb::handle replicaId);

static int64_t DtypeNameToId(std::string_view name)
{
    for (auto const& e : KDtypeIds)
    {
        if (e.name == name) { return e.id; }
    }
    throw std::runtime_error(fmt::format("Unsupported torch dtype name: {}", name));
}

// === DLPack metadata conversion ===

static Tensor::DType DlpackDtypeToDs4(DLDataType const& dt)
{
    if (dt.lanes != 1) { throw std::runtime_error(fmt::format("Unsupported DLPack dtype lanes={}", dt.lanes)); }
    switch (dt.code)
    {
    case kDLFloat:
        switch (dt.bits)
        {
        case 16: return Tensor::DType::Float16;
        case 32: return Tensor::DType::Float32;
        case 64: return Tensor::DType::Float64;
        default: break;
        }
        break;
    case kDLBfloat:
        if (dt.bits == 16) { return Tensor::DType::BFloat16; }
        break;
    case kDLInt:
        switch (dt.bits)
        {
        case 8: return Tensor::DType::Int8;
        case 16: return Tensor::DType::Int16;
        case 32: return Tensor::DType::Int32;
        case 64: return Tensor::DType::Int64;
        default: break;
        }
        break;
    case kDLUInt:
        switch (dt.bits)
        {
        case 8: return Tensor::DType::UInt8;
        case 16: return Tensor::DType::UInt16;
        case 32: return Tensor::DType::UInt32;
        case 64: return Tensor::DType::UInt64;
        default: break;
        }
        break;
    default: break;
    }
    throw std::runtime_error(fmt::format("Unsupported DLPack dtype: code={} bits={}", static_cast<int>(dt.code), dt.bits));
}

static DeviceId DlpackDeviceToId(DLDevice const& d)
{
    char const* prefix = nullptr;
    switch (d.device_type)
    {
    case kDLCPU: prefix = "cpu"; break;
    case kDLCUDA: prefix = "cuda"; break;
    case kDLROCM: prefix = "rocm"; break;
    case kDLMetal: prefix = "mps"; break;
    default: throw std::runtime_error(fmt::format("Unsupported DLPack device type: {}", static_cast<int>(d.device_type)));
    }
    return DeviceId(fmt::format("{}:{}", prefix, d.device_id));
}

// DLPack contiguity check. A null `strides` means row-major contiguous
// by spec; otherwise the strides must match the row-major progression.
static bool IsRowMajorContiguous(DLTensor const& t)
{
    if (t.strides == nullptr) { return true; }
    int64_t expected = 1;
    for (int i = t.ndim - 1; i >= 0; --i)
    {
        if (t.strides[i] != expected) { return false; }
        expected *= t.shape[i];
    }
    return true;
}

static uint64_t DataPtrFromDlTensor(DLTensor const& t)
{
    auto base = reinterpret_cast<uintptr_t>(t.data);    // NOLINT
    return (base + static_cast<uintptr_t>(t.byte_offset));
}

static bool HasElements(DLTensor const& t)
{
    for (int i = 0; i < t.ndim; ++i)
    {
        if (t.shape[i] == 0) { return false; }
    }
    return true;
}

static DlpackCapsule ReadDlpack(nb::handle tensor)
{
    // `.detach()` first: tensors with `requires_grad=True` (i.e. live model
    // parameters mid-training) fail `__dlpack__()` with
    //   BufferError: Can't export tensors that require gradient, use tensor.detach()
    // and the checkpoint save path is exactly that case. `.detach()` returns
    // a grad-free view sharing the same underlying storage — no copy — and
    // is a no-op for tensors that don't require grad. Either way the
    // DLPack capsule points at the original bytes.
    nb::object detached = tensor.attr("detach")();
    auto       device   = nb::cast<nb::tuple>(detached.attr("__dlpack_device__")());
    if (device.size() != 2) { throw std::runtime_error("tensor.__dlpack_device__() returned an invalid device"); }
    auto const deviceType = static_cast<DLDeviceType>(nb::cast<int32_t>(device[0]));
    if (deviceType == kDLCUDA || deviceType == kDLROCM)
    {
        // CUDA's current device is thread-local. Async registration runs on a
        // worker thread, which otherwise starts on device 0 and cannot export
        // tensors owned by the process's assigned device.
        nb::module_::import_("torch.cuda").attr("set_device")(device[1]);
    }
    nb::object cap = detached.attr("__dlpack__")();
    if (cap.is_none()) { throw std::runtime_error("tensor.__dlpack__() returned None"); }

    if (void* p = PyCapsule_GetPointer(cap.ptr(), "dltensor_versioned"); p != nullptr)
    {
        auto* mt = static_cast<DLManagedTensorVersioned*>(p);
        return {.capsule = std::move(cap), .tnsr = &mt->dl_tensor};
    }
    PyErr_Clear();
    if (void* p = PyCapsule_GetPointer(cap.ptr(), "dltensor"); p != nullptr)
    {
        auto* mt = static_cast<DLManagedTensor*>(p);
        return {.capsule = std::move(cap), .tnsr = &mt->dl_tensor};
    }
    throw std::runtime_error("tensor.__dlpack__() returned a PyCapsule with an unexpected name");
}

// === Megatron / torch type predicates ===
//
// Per-leaf `nb::module_::import_(...)` + `attr(...)` adds ~3-5us of overhead
// on a 2k-tensor state_dict (~6-10ms total per save). Cache the type objects
// the first time we see them. Stored as `nb::object*` and intentionally
// leaked to avoid decref-after-Python-finalization at process exit.

static CachedTypes const& CachedTypesGet()
{
    static CachedTypes Types = [] {
        nb::gil_scoped_acquire acquire;
        auto                   torchModule                  = nb::module_::import_("torch");
        auto                   mappingModule                = nb::module_::import_("nvrx_checkpointing.megatron.mapping");
        auto                   megatronShardedTensor        = nb::object(nb::none());
        auto                   megatronShardedTensorFactory = nb::object(nb::none());
        auto                   megatronShardedObject        = nb::object(nb::none());
        auto                   importlibUtil                = nb::module_::import_("importlib.util");
        auto                   findSpec                     = importlibUtil.attr("find_spec");
        if (!findSpec("megatron").is_none() && !findSpec("megatron.core").is_none()
            && !findSpec("megatron.core.dist_checkpointing").is_none() && !findSpec("megatron.core.dist_checkpointing.mapping").is_none())
        {
            auto megatronMappingModule   = nb::module_::import_("megatron.core.dist_checkpointing.mapping");
            megatronShardedTensor        = megatronMappingModule.attr("ShardedTensor");
            megatronShardedTensorFactory = megatronMappingModule.attr("ShardedTensorFactory");
            megatronShardedObject        = megatronMappingModule.attr("ShardedObject");
        }
        return CachedTypes{
            .torchTensor                  = torchModule.attr("Tensor"),
            .torchDtype                   = torchModule.attr("dtype"),
            .torchGenerator               = torchModule.attr("Generator"),
            .localShardedTensor           = mappingModule.attr("ShardedTensor"),
            .localShardedTensorFactory    = mappingModule.attr("ShardedTensorFactory"),
            .localShardedObject           = mappingModule.attr("ShardedObject"),
            .megatronShardedTensor        = std::move(megatronShardedTensor),
            .megatronShardedTensorFactory = std::move(megatronShardedTensorFactory),
            .megatronShardedObject        = std::move(megatronShardedObject),
        };
    }();
    return Types;
}

static bool IsTorchTensor(nb::handle obj)
{ return nb::isinstance(obj, CachedTypesGet().torchTensor); }
static bool IsTorchDtype(nb::handle obj)
{ return nb::isinstance(obj, CachedTypesGet().torchDtype); }
static bool IsTorchGenerator(nb::handle obj)
{ return nb::isinstance(obj, CachedTypesGet().torchGenerator); }
static bool IsMegatronShardedTensor(nb::handle obj)
{
    auto const& types = CachedTypesGet();
    return nb::isinstance(obj, types.localShardedTensor)
           || (!types.megatronShardedTensor.is_none() && nb::isinstance(obj, types.megatronShardedTensor));
}
static bool IsMegatronShardedTensorFactory(nb::handle obj)
{
    auto const& types = CachedTypesGet();
    return nb::isinstance(obj, types.localShardedTensorFactory)
           || (!types.megatronShardedTensorFactory.is_none() && nb::isinstance(obj, types.megatronShardedTensorFactory));
}
static bool IsMegatronShardedObject(nb::handle obj)
{
    auto const& types = CachedTypesGet();
    return nb::isinstance(obj, types.localShardedObject)
           || (!types.megatronShardedObject.is_none() && nb::isinstance(obj, types.megatronShardedObject));
}

// === State extraction ===

static nb::object AddressableTensor(std::string_view name, nb::handle value)
{
    auto owner = nb::borrow<nb::object>(value);
    auto dlp   = ReadDlpack(owner);
    if (DataPtrFromDlTensor(*dlp.tnsr) != 0 || !HasElements(*dlp.tnsr)) { return owner; }

    int const hasDequantize = PyObject_HasAttrString(owner.ptr(), "dequantize");
    if (hasDequantize < 0) { throw nb::python_error(); }
    if (hasDequantize == 0) { throw std::runtime_error(fmt::format("Tensor {} has no addressable storage", name)); }

    // TODO_HOST_DEQUANT: Revisit this for host-based dequantization.
    owner = owner.attr("detach")().attr("dequantize")();
    dlp   = ReadDlpack(owner);
    if (DataPtrFromDlTensor(*dlp.tnsr) == 0 && HasElements(*dlp.tnsr))
    {
        throw std::runtime_error(fmt::format("Tensor {} still has no addressable storage after dequantization", name));
    }
    return owner;
}

static DTensor BareTensorMeta(RankTopology const& topo, std::string_view name, DLTensor const& t)
{
    DTensor        meta{};
    DTensor::Shard shard;
    shard.rank     = topo.rank;
    meta.dtype     = DlpackDtypeToDs4(t.dtype);
    shard.device   = DlpackDeviceToId(t.device);
    shard.handle   = DataPtrFromDlTensor(t);
    uint64_t nelem = 1;
    if (!IsRowMajorContiguous(t)) { throw std::runtime_error(fmt::format("Tensor {} is not contiguous", name)); }
    if (std::cmp_greater(t.ndim, Tensor::MaxDims))
    {
        throw std::runtime_error(fmt::format("Tensor {} has too many dimensions: {}", name, t.ndim));
    }
    if (t.ndim == 0)
    {
        meta.fulldim.push_back(1);
        shard.sharddim.push_back(Tensor::Range{.begin = 0, .end = 1});
    }
    else
    {
        for (int i = 0; i < t.ndim; ++i)
        {
            auto dim = t.shape[i];    // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            if (dim <= 0) { throw std::runtime_error(fmt::format("Tensor {} has non-positive dim[{}]: {}", name, i, dim)); }
            nelem = nelem * static_cast<uint64_t>(dim);
            meta.fulldim.push_back(static_cast<uint64_t>(dim));
            shard.sharddim.push_back(Tensor::Range{.begin = 0, .end = static_cast<uint64_t>(dim)});
        }
    }
    shard.storageSize = nelem * Tensor::GetDTypeSize(meta.dtype);
    shard.AssertShardIsValid(meta.fulldim);
    meta.AddShard(shard);
    return meta;
}

static TensorState TorchTensorState(RankTopology const& topo, std::string_view name, nb::handle value)
{
    auto owner = AddressableTensor(name, value);
    auto dlp   = ReadDlpack(owner);
    return {.meta = BareTensorMeta(topo, name, *dlp.tnsr), .owner = std::move(owner)};
}

static StateNode::Scalar TorchDtypeState(nb::handle value)
{ return StateNode::Scalar{DtypeNameToId(StrView(nb::str(value)))}; }

static StateNode::Blob TorchGeneratorBlob(std::string_view name, nb::handle value)
{
    auto state = value.attr("get_state")();
    // get_state() returns a uint8 CPU tensor on modern torch.
    // Force-CPU defensively in case a future torch changes that.
    auto cpuState = state.attr("cpu")();
    auto dlp      = ReadDlpack(cpuState);

    if (dlp.tnsr->device.device_type != kDLCPU)
    {
        throw std::runtime_error(fmt::format("Generator state for {} did not land on CPU after .cpu()", name));
    }
    if (dlp.tnsr->ndim != 1)
    {
        throw std::runtime_error(fmt::format("Generator state for {} has unexpected ndim={}", name, dlp.tnsr->ndim));
    }
    if (!IsRowMajorContiguous(*dlp.tnsr)) { throw std::runtime_error(fmt::format("Generator state for {} is not contiguous", name)); }

    auto              nbytes = static_cast<size_t>(dlp.tnsr->shape[0]) * static_cast<size_t>((dlp.tnsr->dtype.bits + 7U) / 8U)    // NOLINT
                               * static_cast<size_t>(dlp.tnsr->dtype.lanes);
    auto const*       src    = reinterpret_cast<char const*>(DataPtrFromDlTensor(*dlp.tnsr));    // NOLINT
    std::vector<char> data(nbytes);
    std::memcpy(data.data(), src, nbytes);
    return data;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static TensorState ShardedTensorState(RankTopology const& topo, nb::str& name, nb::dict const& container)
{
    using Range = Tensor::Range;
    DTensor ds4tensor{};

    auto shtKey     = nb::cast<nb::str>(container["key"]);
    auto nameView   = StrView(name);
    auto shtKeyView = StrView(shtKey);
    if (!nameView.starts_with(shtKeyView))
    {
        // Match the prior C++ walker's TODO_SHTEN_KEY policy: when
        // the dict-path name doesn't match the DTensor's own
        // `key`, the sharded key wins.
        name     = shtKey;
        nameView = shtKeyView;
    }

    struct ShardedTensorError : public std::runtime_error
    {
        ShardedTensorError(nb::str const& key, nb::dict const& container, std::string_view const& message) :
            std::runtime_error(
                fmt::format("Error Registering StateDict DTensor::{} key = {} sharded_tensor = {}", message, key, nb::str(container)))
        {}
    };
    auto throwError = [&](std::string_view const& message) { throw ShardedTensorError(shtKey, container, message); };

    auto owner = AddressableTensor(shtKeyView, container["data"]);
    auto dlp   = ReadDlpack(owner);
    if (!IsRowMajorContiguous(*dlp.tnsr)) { throw std::runtime_error(fmt::format("Tensor {} is not contiguous", shtKeyView)); }
    ds4tensor.dtype = DlpackDtypeToDs4(dlp.tnsr->dtype);
    DTensor::Shard shard{};
    shard.rank   = topo.rank;
    shard.handle = DataPtrFromDlTensor(*dlp.tnsr);
    shard.device = DlpackDeviceToId(dlp.tnsr->device);

    auto globalShape    = nb::cast<nb::tuple>(container["global_shape"]);
    auto localShape     = nb::cast<nb::tuple>(container["local_shape"]);
    auto globalOffset   = nb::cast<nb::tuple>(container["global_offset"]);
    auto flattenedRange = container["flattened_range"];

    if (globalShape.size() != globalOffset.size())
    {
        throwError(
            fmt::format("global_shape size ([{}]) mismatch with global_offset size ([{}])", nb::str(globalShape), nb::str(globalOffset)));
    }

    auto prependAxisNumValue = [&] -> size_t {
        auto prependAxisNum = container["prepend_axis_num"];
        if (prependAxisNum.is_none() && globalShape.size() == localShape.size()) { return 0; }
        if (prependAxisNum.is_none() && globalShape.size() != localShape.size())
        {
            throwError(fmt::format("prepend_axis_num is None inconsistent with global_shape([{}]), local_shape([{}])",
                                   nb::str(globalShape),
                                   nb::str(localShape)));
        }
        if (!nb::isinstance<int64_t>(prependAxisNum)) { throwError("prependAxisNum is not int"); }
        auto value = nb::cast<int64_t>(prependAxisNum);
        if (value >= 0 && globalShape.size() >= localShape.size() && std::cmp_equal(value, globalShape.size() - localShape.size()))
        {
            return static_cast<size_t>(value);
        }
        throwError(fmt::format("prepend_axis_num = {} inconsistent with global_shape([{}]), local_shape([{}])",
                               value,
                               nb::str(globalShape),
                               nb::str(localShape)));
        std::unreachable();
    }();

    [&] {
        auto allowShapeMismatch = container["allow_shape_mismatch"];
        if (allowShapeMismatch.is_none()) { return; }
        if (!nb::isinstance<bool>(allowShapeMismatch)) { throwError("allow_shape_mismatch is not bool"); }
        if (!nb::cast<bool>(allowShapeMismatch)) { return; }
#ifdef TODO_ALLOW_SHAPE_MISMATCH
        fmt::print(stderr,
                   "[native_ckpt:cli][TODO][StateDict::Register::DTensor] key = {}, "
                   "allow_shape_mismatch = {} globalShape = [{}] localShape=[{}] globalOffset=[{}]\n",
                   shtKeyView,
                   true,
                   nb::str(globalShape),
                   nb::str(localShape),
                   nb::str(globalOffset));
#endif
    }();

    if (!flattenedRange.is_none()) { throwError("flattened_range is not None"); }

    [&] {
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
            if (i < prependAxisNumValue)
            {
                if (fragVal != globalDimVal)
                {
                    throwError(fmt::format("axis_fragmentations[{}] does not match global_shape[{}] for prepended axis", i, i));
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

    for (auto const& val : globalShape) { ds4tensor.fulldim.push_back(nb::cast<uint64_t>(val)); }
    shard.storageSize = 1;
    for (size_t i = 0; i < globalShape.size(); ++i)
    {
        auto    start = nb::cast<int64_t>(globalOffset[i]);
        int64_t end{};
        if (i < prependAxisNumValue) { end = start + 1; }
        else
        {
            end = start + nb::cast<int64_t>(localShape[i - prependAxisNumValue]);
        }
        shard.sharddim.push_back(Range{.begin = static_cast<size_t>(start), .end = static_cast<size_t>(end)});
        shard.storageSize *= (end - start);
    }
    shard.storageSize = shard.storageSize * Tensor::GetDTypeSize(ds4tensor.dtype);

    size_t dlpackBytes = static_cast<size_t>((dlp.tnsr->dtype.bits + 7U) / 8U) * dlp.tnsr->dtype.lanes;    // NOLINT
    for (int i = 0; i < dlp.tnsr->ndim; ++i)
    {
        dlpackBytes *= static_cast<size_t>(dlp.tnsr->shape[i]);    // NOLINT
    }
    if (shard.storageSize != dlpackBytes)
    {
        throw std::runtime_error(fmt::format(
            "Tensor {} size mismatch: sharddim implies {} bytes, DLPack shape implies {}", shtKeyView, shard.storageSize, dlpackBytes));
    }
    ds4tensor.AddShard(shard);
    // ds4tensor.AssertShardIsValid();
    return {.meta = ds4tensor, .owner = std::move(owner)};
}

static StateNode::Blob ShardedObjectBlob(nb::handle value)
{
    static constexpr char const* PreparedBlobAttr = "_nvrx_serialized_blob";
    auto const                   hasPreparedBlob  = PyObject_HasAttrString(value.ptr(), PreparedBlobAttr);
    if (hasPreparedBlob < 0) { throw nb::python_error(); }
    if (hasPreparedBlob != 0)
    {
        auto const raw = nb::cast<nb::bytes>(value.attr(PreparedBlobAttr));
        return {raw.c_str(), raw.c_str() + raw.size()};
    }

    nb::object io    = nb::module_::import_("io");
    nb::object torch = nb::module_::import_("torch");
    nb::object buf   = io.attr("BytesIO")();
    nb::list   one;
    one.append(value.attr("data"));
    torch.attr("save")(one, buf);
    auto const raw = nb::cast<nb::bytes>(buf.attr("getvalue")());
    return {raw.c_str(), raw.c_str() + raw.size()};
}

static StateNode&
RegisterTensorState(PyStateDictAccessor& scan, StateNode::Handle parent, std::string_view name, DTensor tensor, nb::handle owner)
{
    auto& node = name.contains('.') ? scan.RegisterStateByFQN(name, tensor) : scan.RegisterState(parent, name, tensor);
    scan.AcquireTensorObjectRefForSnapshot(node.handle, nb::borrow<nb::object>(owner));
    return node;
}

static void RegisterShardedTorchTensorFactory(PyStateDictAccessor& scan, StateNode::Handle node, nb::str const& name, nb::handle factory)
{
    auto factoryKey     = nb::cast<nb::str>(factory.attr("key"));
    auto factoryKeyView = StrView(factoryKey);
    auto nameView       = StrView(name);
    if (factoryKeyView != nameView && !factoryKeyView.contains('.'))
    {
        throw std::runtime_error(fmt::format("ShardedTensorFactory key mismatch: expected {}, got {}", nameView, factoryKeyView));
    }
    auto buildFunc      = factory.attr("build");
    auto shardedTensors = buildFunc();
    if (!nb::isinstance<nb::list>(shardedTensors))
    {
        throw std::runtime_error(fmt::format("Expected ShardedTensorFactory::build to return a list of ShardedTensors for key: {}", name));
    }

    auto shardedTensorsList = nb::cast<nb::list>(shardedTensors);
    for (auto const& item : shardedTensorsList)
    {
        if (!IsMegatronShardedTensor(item))
        {
            throw std::runtime_error(fmt::format("Expected ShardedTensorFactory::build to return a list of ShardedTensors for "
                                                 "key: {} but got {}",
                                                 name,
                                                 nb::str(item.type())));
        }
        auto innerContainer = nb::cast<nb::dict>(item.attr("__dict__"));
        auto shtKey         = nb::cast<nb::str>(innerContainer["key"]);
        auto shtKeyView     = StrView(shtKey);
        // Verify the inner key is a strict child of (or equal to) the
        // factory's fqn. Root parent + dot-less name gives just the
        // name; the dotted-name branch is for callers that already
        // pre-prefix the path.
        std::string fqn;
        if (factoryKeyView != nameView) { fqn = std::string(factoryKeyView); }
        else if (nameView.contains('.')) { fqn = std::string(nameView); }
        else
        {
            auto parentFqn = scan.GetNodeId(node);
            fqn            = parentFqn.empty() ? std::string(nameView) : fmt::format("{}.{}", parentFqn, nameView);
        }
        if (shtKeyView != fqn && (!shtKeyView.starts_with(fqn) || shtKeyView.size() <= fqn.size() || shtKeyView[fqn.size()] != '.'))
        {
            throw std::runtime_error(fmt::format("ShardedTensorFactory::build returned DTensor with key {} that is not a child of "
                                                 "the factory obj fqn {}",
                                                 shtKeyView,
                                                 fqn));
        }
        // TODO_NOREPLICAID : # Remove reliance on replica_id to prune state-dict/snapshotted tensors
        if (!IsMainReplica(innerContainer["replica_id"])) { continue; }
        auto tensorName  = shtKey;
        auto tensorState = ShardedTensorState(scan.GetRankTopo(), tensorName, innerContainer);
        RegisterTensorState(scan, node, StrView(tensorName), tensorState.meta, tensorState.owner);
    }
}

// Only main replicas own checkpoint payloads.
// TODO_NOREPLICAID : # Remove reliance on replica_id to prune state-dict/snapshotted tensors
static bool IsMainReplica(nb::handle replicaId)
{
    if (replicaId.is_none()) { return true; }
    if (nb::isinstance<nb::int_>(replicaId)) { return nb::cast<int64_t>(replicaId) == 0; }
    if (nb::isinstance<nb::tuple>(replicaId))
    {
        auto tuple = nb::cast<nb::tuple>(replicaId);
        return std::ranges::all_of(tuple, [](auto const& v) { return nb::cast<int64_t>(v) == 0; });
    }
    if (nb::isinstance<nb::list>(replicaId))
    {
        auto list = nb::cast<nb::list>(replicaId);
        return std::ranges::all_of(list, [](auto const& v) { return nb::cast<int64_t>(v) == 0; });
    }
    throw std::runtime_error("Unrecognized replica_id type (expected None, int, or tuple/list of int)");
}

static void RegisterShardedObject(PyStateDictAccessor& scan, nb::handle val)
{
    // Main-replica objects are serialized fresh for each save as DCP byte entries.
    if (!IsMainReplica(val.attr("replica_id"))) { return; }
    auto uniqueKey = nb::cast<std::string>(val.attr("unique_key"));
    scan.RegisterStateByFQN(uniqueKey, ShardedObjectBlob(val));
}

// After `apply_factories`, a key like `decoder.layers.0.mixer.in_proj.weight`
// gets replaced with `[DTensor(...weight.z), DTensor(...weight.x),
// DTensor(...weight.B), DTensor(...weight.C), DTensor(...weight.dt)]`.
// Register each main-replica element using its own `.key` as the FQN — same shape as nvrx's
// post-`save_preprocess` state_dict, so the cross-provider tensor compare can
// match tensors per-FQN. Returns false when the list isn't an all-DTensor
// list so the generic walker can fall through to its existing per-element-type
// handling.
bool TryRegisterTorchStateDictList(PyStateDictAccessor& scan, StateNode::Handle node, nb::str const& key, nb::list const& listObj)
{
    if (listObj.empty()) { return false; }
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
    auto const parentPath = scan.GetNodeId(node);
    auto const fqn = (nameView.contains('.') || parentPath.empty()) ? std::string(nameView) : fmt::format("{}.{}", parentPath, nameView);
    for (auto const& item : listObj)
    {
        auto itemContainer = nb::cast<nb::dict>(item.attr("__dict__"));
        auto shtKey        = nb::cast<nb::str>(itemContainer["key"]);
        auto shtKeyView    = StrView(shtKey);

        if (shtKeyView != fqn && (!shtKeyView.starts_with(fqn) || shtKeyView.size() <= fqn.size() || shtKeyView[fqn.size()] != '.'))
        {
            // TODO_BREAK_BACK_COMPAT The following changes in megatron/core/ssm/mamba_mixer.py:1348
            // for back-compat are forcing a weakening of the strong change
            //  conv_checkpoint_key_map = {"conv1d_weight": "conv1d.weight", "conv1d_bias": "conv1d.bias"}
            //  for param_name, checkpoint_name in conv_checkpoint_key_map.items():
            //      sharded_state_dict[f"{prefix}{param_name}"].key = f"{prefix}{checkpoint_name}"

            // throw std::runtime_error("DTensor list element key '{}' is not a child of state-dict key '{}'", shtKeyView, fqn);
        }
        // TODO_NOREPLICAID : # Remove reliance on replica_id to prune state-dict/snapshotted tensors
        if (!IsMainReplica(itemContainer["replica_id"])) { continue; }
        auto tensorName  = shtKey;
        auto tensorState = ShardedTensorState(scan.GetRankTopo(), tensorName, itemContainer);
        RegisterTensorState(scan, node, StrView(tensorName), tensorState.meta, tensorState.owner);
    }
    return true;
}

bool TryRegisterTorchStateDict(PyStateDictAccessor& scan, StateNode::Handle node, nb::str const& key, nb::handle val)
{
    if (IsTorchTensor(val))
    {
        auto tensorState = TorchTensorState(scan.GetRankTopo(), StrView(key), val);
        RegisterTensorState(scan, node, StrView(key), tensorState.meta, tensorState.owner);
        return true;
    }
    if (IsTorchDtype(val))
    {
        scan.RegisterState(node, StrView(key), TorchDtypeState(val));
        return true;
    }
    if (IsMegatronShardedTensor(val))
    {
        // TODO_NOREPLICAID : # Remove reliance on replica_id to prune state-dict/snapshotted tensors
        if (!IsMainReplica(val.attr("replica_id"))) { return true; }
        auto tensorName  = key;
        auto tensorState = ShardedTensorState(scan.GetRankTopo(), tensorName, nb::cast<nb::dict>(val.attr("__dict__")));
        RegisterTensorState(scan, node, StrView(tensorName), tensorState.meta, tensorState.owner);
        return true;
    }
    if (IsMegatronShardedTensorFactory(val))
    {
        RegisterShardedTorchTensorFactory(scan, node, key, val);
        return true;
    }
    if (IsMegatronShardedObject(val))
    {
        // The object's unique_key is a full FQN, so it reparents to the state-dict root.
        RegisterShardedObject(scan, val);
        return true;
    }
    if (IsTorchGenerator(val))
    {
        scan.RegisterState(node, StrView(key), TorchGeneratorBlob(StrView(key), val));
        return true;
    }
    return false;
}
