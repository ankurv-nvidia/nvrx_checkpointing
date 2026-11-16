
#include "binding.h"
#include "checkpoint.h"
#include "distcp.h"
#include "messages.h"
#include "state.h"
#include "tensor.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <future>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <tuple>

namespace nb   = nanobind;
namespace cipc = checkpoint::ipc;

namespace
{
nb::dict StatePythonTypes()
{
    nb::dict   types;
    auto const torch          = nb::module_::import_("torch");
    auto const numpy          = nb::module_::import_("numpy");
    types["torch"]            = torch;
    types["numpy"]            = numpy;
    types["tensor"]           = torch.attr("Tensor");
    types["enum"]             = nb::module_::import_("enum").attr("Enum");
    types["namespace"]        = nb::module_::import_("argparse").attr("Namespace");
    types["simple_namespace"] = nb::module_::import_("types").attr("SimpleNamespace");
    types["array"]            = numpy.attr("ndarray");
    types["numpy_scalar"]     = numpy.attr("generic");
    types["generator"]        = torch.attr("Generator");
    types["dtype"]            = torch.attr("dtype");
    types["device"]           = torch.attr("device");
    types["mapping"]          = nb::module_::import_("collections.abc").attr("Mapping");
    types["dtensor"]          = nb::none();
    types["shape_offset"]     = nb::none();
    if (nb::cast<bool>(nb::module_::import_("torch.distributed").attr("is_available")()))
    {
        types["dtensor"]      = nb::module_::import_("torch.distributed.tensor").attr("DTensor");
        types["shape_offset"] = nb::module_::import_("torch.distributed.tensor._utils").attr("compute_local_shape_and_global_offset");
    }
    return types;
}

bool StateIsNamespace(nb::handle value, nb::dict const& types)
{ return nb::isinstance(value, types["namespace"]) || nb::isinstance(value, types["simple_namespace"]); }

std::string StatePath(std::string_view parent, nb::handle key)
{
    auto name = Str(nb::str(key));
    if (name.empty()) { throw std::invalid_argument("Native state keys must have nonempty addresses"); }
    return parent.empty() ? name : fmt::format("{}.{}", parent, name);
}

std::optional<nb::dict> StateMapping(nb::handle value, nb::dict const& types)
{
    if (nb::isinstance<nb::dict>(value)) { return nb::cast<nb::dict>(value); }
    if (StateIsNamespace(value, types)) { return nb::cast<nb::dict>(value.attr("__dict__")); }
    return std::nullopt;
}

bool StateIsContainer(nb::handle value, nb::dict const& types)
{
    return nb::isinstance<nb::dict>(value) || StateIsNamespace(value, types) || nb::isinstance<nb::list>(value)
           || nb::isinstance<nb::tuple>(value);
}

static StateNode::Scalar ScalarState(nb::handle obj)
{
    if (nb::isinstance<bool>(obj)) { return StateNode::Scalar{nb::cast<bool>(obj)}; }
    if (nb::isinstance<nb::int_>(obj))
    {
        int        overflow    = 0;
        auto const signedValue = PyLong_AsLongLongAndOverflow(obj.ptr(), &overflow);
        if (PyErr_Occurred() != nullptr) { throw nb::python_error(); }
        if (overflow == 0) { return StateNode::Scalar{static_cast<int64_t>(signedValue)}; }
        if (overflow < 0) { throw std::out_of_range("Checkpoint integer is below the signed 64-bit range"); }
        auto const unsignedValue = PyLong_AsUnsignedLongLong(obj.ptr());
        if (PyErr_Occurred() != nullptr) { throw nb::python_error(); }
        return StateNode::Scalar{static_cast<uint64_t>(unsignedValue)};
    }
    if (nb::isinstance<nb::float_>(obj)) { return StateNode::Scalar{nb::cast<double>(obj)}; }
    throw std::out_of_range("Conversion of nanobind::object to Scalar failed");
}

class StateVisit
{
    public:
    StateVisit(std::unordered_set<PyObject*>& active, nb::handle value) : _active(active), _value(value.ptr())
    {
        if (!_active.insert(_value).second) { throw std::invalid_argument("Native state contains a recursive container"); }
    }
    ~StateVisit() { _active.erase(_value); }
    CLASS_DELETE_COPY_AND_MOVE(StateVisit);

    private:
    std::unordered_set<PyObject*>& _active;
    PyObject*                      _value;
};

nb::object StateTuple(nb::handle target, nb::list const& values)
{
    if (!nb::isinstance<nb::tuple>(target)) { return nb::tuple(values); }
    return nb::hasattr(target, "_fields") ? nb::borrow<nb::object>(target.type())(*values) : nb::borrow<nb::object>(target.type())(values);
}

template <typename Visitor>
nb::object WalkState(nb::handle                     value,
                     std::string const&             path,
                     nb::dict const&                types,
                     std::unordered_set<PyObject*>& active,
                     Visitor const&                 visit,
                     size_t                         depth = 0)
{
    if (depth > 100) { throw std::invalid_argument("Native checkpoint nesting exceeds the supported depth"); }
    if (auto leaf = visit(value, path)) { return std::move(*leaf); }
    StateVisit guard(active, value);
    if (auto mapping = StateMapping(value, types))
    {
        for (auto key : mapping->keys())
        {
            auto old   = nb::borrow<nb::object>((*mapping)[key]);
            auto child = WalkState(old, StatePath(path, key), types, active, visit, depth + 1);
            if (!child.is(old)) { (*mapping)[key] = child; }
        }
        return nb::borrow<nb::object>(value);
    }
    nb::list children;
    bool     changed = false;
    size_t   index   = 0;
    for (auto old : nb::cast<nb::sequence>(value))
    {
        auto child = WalkState(old, fmt::format("{}.{}", path, index), types, active, visit, depth + 1);
        changed    = changed || !child.is(old);
        children.append(child);
        ++index;
    }
    if (!changed) { return nb::borrow<nb::object>(value); }
    if (nb::isinstance<nb::tuple>(value)) { return StateTuple(value, children); }
    for (size_t offset = 0; offset < children.size(); ++offset) { value[offset] = children[offset]; }
    return nb::borrow<nb::object>(value);
}
}    // namespace

namespace checkpoint::python
{
namespace nb = nanobind;

class NativeStateRegistration
{
    static constexpr size_t MaxNestingDepth = 100;
    using Buffer                            = std::vector<char>;

    enum class Kind : uint8_t
    {
        None,
        Int,
        UInt,
        Float,
        Bool,
        String,
        Bytes,
        List,
        Tuple,
        Dict,
        Namespace,
        Array,
        Dtype,
        Enum,
        Device,
        EmptyTensor,
        Generator,
    };

    struct WriteItem
    {
        std::string           path;
        bool                  tensor;
        std::vector<uint64_t> offsets;
        uint64_t              cost;
    };

    public:
    explicit NativeStateRegistration(PyStateDictAccessor& scan) : _scan(scan), _types(StatePythonTypes()) {}

    void Register(nb::dict const& state)
    {
        CollectMappingWriteItems({}, state, 0);
        WalkState(state, {}, _types, _active, [&](nb::handle value, std::string const& path) -> std::optional<nb::object> {
            if (path.empty()) { return std::nullopt; }
            return RegisterValue(path, value);
        });
        Buffer marker;
        Append(marker, "NVRXNST2");
        Put32(marker, _records.size());
        for (auto const& [path, kind] : _records)
        {
            PutString(marker, path);
            Put8(marker, kind);
        }
        Put32(marker, _writeItems.size());
        for (auto const& item : _writeItems)
        {
            PutString(marker, item.path);
            Put8(marker, item.tensor ? 1 : 0);
            if (item.tensor)
            {
                Put32(marker, item.offsets.size());
                for (auto const offset : item.offsets) { Put64(marker, offset); }
            }
            Put64(marker, item.cost);
        }
        RegisterBlob(fmt::format("__nvrx_native_state_rank_{}__", _scan.GetRankTopo().rank), std::move(marker));
    }

    private:
    static void Put8(Buffer& output, uint8_t value) { output.push_back(static_cast<char>(value)); }

    static void Put32(Buffer& output, size_t value)
    {
        if (value > std::numeric_limits<uint32_t>::max()) { throw std::length_error("Native state count exceeds its wire limit"); }
        for (unsigned int shift = 0; shift < 32; shift += 8) { Put8(output, static_cast<uint8_t>(value >> shift)); }
    }

    static void Put64(Buffer& output, uint64_t value)
    {
        for (unsigned int shift = 0; shift < 64; shift += 8) { Put8(output, static_cast<uint8_t>(value >> shift)); }
    }

    static void Append(Buffer& output, std::string_view value) { output.insert(output.end(), value.begin(), value.end()); }

    static void PutString(Buffer& output, std::string_view value)
    {
        Put32(output, value.size());
        Append(output, value);
    }

    static void PutByteString(Buffer& output, nb::bytes const& value)
    {
        Put64(output, value.size());
        Append(output, {value.c_str(), value.size()});
    }

    static std::string DtypeName(nb::handle value)
    {
        auto name = Str(nb::str(value));
        if (name.starts_with("torch.")) { name.erase(0, 6); }
        return name;
    }

    bool IsNamespace(nb::handle value) const { return StateIsNamespace(value, _types); }

    bool ContainsTensor(nb::handle value, size_t depth = 0)
    {
        bool found = false;
        WalkState(
            value,
            {},
            _types,
            _active,
            [&](nb::handle child, std::string const&) -> std::optional<nb::object> {
                found = found || nb::isinstance(child, _types["tensor"]);
                if (found || !StateIsContainer(child, _types)) { return nb::borrow<nb::object>(child); }
                return std::nullopt;
            },
            depth);
        return found;
    }

    bool IsTerminalWriteItem(nb::handle value, size_t depth)
    {
        if (depth > MaxNestingDepth) { throw std::invalid_argument("Native checkpoint nesting exceeds the supported depth"); }
        if (nb::isinstance(value, _types["mapping"])) { return false; }
        if (!nb::isinstance<nb::list>(value)) { return true; }
        StateVisit guard(_writeActive, value);
        for (auto const child : nb::cast<nb::list>(value))
        {
            if ((nb::isinstance(child, _types["mapping"]) || nb::isinstance<nb::list>(child)) && !IsTerminalWriteItem(child, depth + 1))
            {
                return false;
            }
            if (nb::isinstance(child, _types["tensor"])) { return false; }
        }
        return true;
    }

    void CollectMappingWriteItems(std::string_view path, nb::handle value, size_t depth)
    {
        StateVisit guard(_writeActive, value);
        auto const items = value.attr("items")();
        for (auto const item : nb::cast<nb::iterable>(items))
        {
            auto const pair = nb::cast<nb::tuple>(item);
            CollectWriteItems(StatePath(path, pair[0]), pair[1], depth);
        }
    }

    void CollectWriteItems(std::string const& path, nb::handle value, size_t depth)
    {
        if (depth > MaxNestingDepth) { throw std::invalid_argument("Native checkpoint nesting exceeds the supported depth"); }
        if (nb::isinstance(value, _types["mapping"]))
        {
            CollectMappingWriteItems(path, value, depth + 1);
            return;
        }
        if (!IsTerminalWriteItem(value, depth))
        {
            StateVisit guard(_writeActive, value);
            size_t     index = 0;
            for (auto const child : nb::cast<nb::iterable>(value))
            {
                CollectWriteItems(fmt::format("{}.{}", path, index), child, depth + 1);
                ++index;
            }
            return;
        }
        if (!_writePaths.insert(path).second)
        {
            throw std::invalid_argument(fmt::format("Native checkpoint addresses overlap at '{}'", path));
        }
        if (!nb::isinstance(value, _types["tensor"]))
        {
            _writeItems.push_back({path, false, {}, 1});
            return;
        }
        auto const count = nb::cast<uint64_t>(value.attr("numel")());
        auto const width = nb::cast<uint64_t>(value.attr("element_size")());
        if (width != 0 && count > std::numeric_limits<uint64_t>::max() / width)
        {
            throw std::overflow_error("Native checkpoint tensor write cost exceeds the supported range");
        }
        auto const cost = std::max<uint64_t>(count * width, 1);
        if (!_types["dtensor"].is_none() && nb::isinstance(value, _types["dtensor"]))
        {
            if (value.attr("device_mesh").attr("get_coordinate")().is_none()) { return; }
            nb::object const local = value.attr("_local_tensor");
            if (nb::hasattr(local, "__create_write_items__"))
            {
                if (nb::cast<uint64_t>(local.attr("numel")()) == 0) { return; }
                for (auto const chunk : nb::cast<nb::iterable>(value.attr("__create_chunk_list__")()))
                {
                    AddTensorWriteItem(path, chunk.attr("offsets"), cost);
                }
            }
            else
            {
                auto const geometry = _types["shape_offset"](value.attr("shape"), value.attr("device_mesh"), value.attr("placements"));
                AddTensorWriteItem(path, geometry[1], cost);
            }
            return;
        }
        nb::object const shape = value.attr("shape");
        _writeItems.push_back({path, true, std::vector<uint64_t>(nb::len(shape), 0), cost});
    }

    void AddTensorWriteItem(std::string const& path, nb::handle offsets, uint64_t cost)
    {
        WriteItem item{path, true, {}, cost};
        for (auto const offset : nb::cast<nb::iterable>(offsets)) { item.offsets.push_back(nb::cast<uint64_t>(offset)); }
        _writeItems.push_back(std::move(item));
    }

    void Encode(Buffer& output, nb::handle value, size_t depth = 0)
    {
        if (depth > MaxNestingDepth) { throw std::invalid_argument("Native checkpoint nesting exceeds the supported depth"); }
        if (value.is_none()) { Put8(output, static_cast<uint8_t>(Kind::None)); }
        else if (nb::isinstance(value, _types["enum"]))
        {
            Put8(output, static_cast<uint8_t>(Kind::Enum));
            Encode(output, value.attr("value"), depth + 1);
        }
        else if (nb::isinstance<bool>(value))
        {
            Put8(output, static_cast<uint8_t>(Kind::Bool));
            Put8(output, nb::cast<bool>(value) ? 1 : 0);
        }
        else if (nb::isinstance<nb::int_>(value))
        {
            auto const integer     = ScalarState(value);
            bool const signedValue = Primitives64Bit::Type::IsSigned(integer.GetType());
            Put8(output, static_cast<uint8_t>(signedValue ? Kind::Int : Kind::UInt));
            Put64(output, signedValue ? std::bit_cast<uint64_t>(integer.Cast<int64_t>()) : integer.Cast<uint64_t>());
        }
        else if (nb::isinstance<nb::float_>(value))
        {
            Put8(output, static_cast<uint8_t>(Kind::Float));
            Put64(output, std::bit_cast<uint64_t>(nb::cast<double>(value)));
        }
        else if (nb::isinstance<nb::str>(value))
        {
            Put8(output, static_cast<uint8_t>(Kind::String));
            PutString(output, StrView(nb::cast<nb::str>(value)));
        }
        else if (nb::isinstance<nb::bytes>(value))
        {
            Put8(output, static_cast<uint8_t>(Kind::Bytes));
            PutByteString(output, nb::cast<nb::bytes>(value));
        }
        else if (nb::isinstance<nb::list>(value) || nb::isinstance<nb::tuple>(value))
        {
            StateVisit guard(_active, value);
            Put8(output, static_cast<uint8_t>(nb::isinstance<nb::list>(value) ? Kind::List : Kind::Tuple));
            auto const sequence = nb::cast<nb::sequence>(value);
            Put32(output, nb::len(sequence));
            for (auto const child : sequence) { Encode(output, child, depth + 1); }
        }
        else if (nb::isinstance<nb::dict>(value) || IsNamespace(value))
        {
            StateVisit guard(_active, value);
            auto const isNamespace = IsNamespace(value);
            auto const mapping     = isNamespace ? nb::cast<nb::dict>(value.attr("__dict__")) : nb::cast<nb::dict>(value);
            Put8(output, static_cast<uint8_t>(isNamespace ? Kind::Namespace : Kind::Dict));
            Put32(output, mapping.size());
            for (auto const& [key, child] : mapping)
            {
                Encode(output, key, depth + 1);
                Encode(output, child, depth + 1);
            }
        }
        else if (nb::isinstance(value, _types["array"]))
        {
            nb::object const dtype = value.attr("dtype");
            if (nb::cast<bool>(dtype.attr("hasobject")) || !dtype.attr("fields").is_none())
            {
                throw std::invalid_argument("Native state supports NumPy arrays with primitive dtypes");
            }
            auto const array = _types["numpy"].attr("ascontiguousarray")(value);
            Put8(output, static_cast<uint8_t>(Kind::Array));
            PutString(output, Str(nb::str(dtype)));
            PutShape(output, value.attr("shape"));
            PutByteString(output, nb::cast<nb::bytes>(array.attr("tobytes")()));
        }
        else if (nb::isinstance(value, _types["numpy_scalar"]))
        {
            auto const item = value.attr("item")();
            if (nb::isinstance(item, _types["numpy_scalar"]))
            {
                throw std::invalid_argument("Unsupported native checkpoint value type: NumPy scalar without a Python equivalent");
            }
            Encode(output, item, depth);
        }
        else if (nb::isinstance(value, _types["dtype"]))
        {
            Put8(output, static_cast<uint8_t>(Kind::Dtype));
            PutString(output, DtypeName(value));
        }
        else if (nb::isinstance(value, _types["device"]))
        {
            Put8(output, static_cast<uint8_t>(Kind::Device));
            PutString(output, Str(nb::str(value)));
        }
        else if (nb::isinstance(value, _types["generator"]))
        {
            nb::object const device = value.attr("device");
            Put8(output, static_cast<uint8_t>(Kind::Generator));
            PutString(output, Str(nb::str(device)));
            auto const bytes = value.attr("get_state")().attr("cpu")().attr("numpy")().attr("tobytes")();
            PutByteString(output, nb::cast<nb::bytes>(bytes));
        }
        else if (nb::isinstance(value, _types["tensor"]) && nb::cast<uint64_t>(value.attr("numel")()) == 0)
        {
            Put8(output, static_cast<uint8_t>(Kind::EmptyTensor));
            PutString(output, DtypeName(value.attr("dtype")));
            PutShape(output, value.attr("shape"));
        }
        else
        {
            throw std::invalid_argument(fmt::format("Unsupported native checkpoint value type: {}", nb::str(value.type())));
        }
    }

    static void PutShape(Buffer& output, nb::handle shape)
    {
        auto const dimensions = nb::cast<nb::sequence>(shape);
        Put32(output, nb::len(dimensions));
        for (auto const size : dimensions) { Put64(output, nb::cast<uint64_t>(size)); }
    }

    void ReservePath(std::string const& path)
    {
        if (path.starts_with("__nvrx_native_state_rank_") || path.starts_with("__nvrx_native_dict_rank_"))
        {
            throw std::invalid_argument("Native checkpoint state uses a reserved metadata address");
        }
        for (auto position = path.find('.'); position != std::string::npos; position = path.find('.', position + 1))
        {
            if (_paths.contains(path.substr(0, position)))
            {
                throw std::invalid_argument(fmt::format("Native checkpoint addresses overlap at '{}'", path));
            }
        }
        auto const childPrefix = path + '.';
        auto const child       = _paths.lower_bound(childPrefix);
        if ((child != _paths.end() && child->starts_with(childPrefix)) || !_paths.insert(path).second)
        {
            throw std::invalid_argument(fmt::format("Native checkpoint addresses overlap at '{}'", path));
        }
    }

    void RegisterBlob(std::string_view path, Buffer bytes)
    {
        auto const size   = bytes.size();
        auto const handle = reinterpret_cast<uint64_t>(bytes.data());
        auto*      node   = _scan.RegisterStateByFQN(
            path, StateNode::DynamicSizeBlob{.replicas = {_scan.GetRankTopo().rank}, .handle = handle, .storageSize = size});
        if (node != nullptr) { _scan.RetainBlobData(std::move(bytes)); }
    }

    std::optional<nb::object> RegisterValue(std::string const& path, nb::handle value)
    {
        if (nb::isinstance(value, _types["tensor"]))
        {
            ReservePath(path);
            if ((!_types["dtensor"].is_none() && nb::isinstance(value, _types["dtensor"]))
                || nb::cast<uint64_t>(value.attr("numel")()) != 0)
            {
                if (!TryRegisterTorchStateDict(_scan, StateNode::Handle::Root, nb::str(path.c_str(), path.size()), value))
                {
                    throw std::invalid_argument(fmt::format("Native tensor '{}' cannot be registered", path));
                }
                return nb::borrow<nb::object>(value);
            }
        }
        else if (StateIsContainer(value, _types) && ContainsTensor(value)) { return std::nullopt; }
        else
        {
            ReservePath(path);
        }

        if (!nb::isinstance(value, _types["enum"])
            && (nb::isinstance<bool>(value) || nb::isinstance<nb::int_>(value) || nb::isinstance<nb::float_>(value)))
        {
            auto       scalar = ScalarState(value);
            auto const type   = scalar.GetType();
            auto const kind   = Primitives64Bit::Type::IsBoolean(type)  ? Kind::Bool
                                : Primitives64Bit::Type::IsFloat(type)  ? Kind::Float
                                : Primitives64Bit::Type::IsSigned(type) ? Kind::Int
                                                                        : Kind::UInt;
            _scan.RegisterStateByFQN(path, std::move(scalar));
            _records.emplace_back(path, static_cast<uint8_t>(kind));
        }
        else
        {
            Buffer bytes;
            Append(bytes, "NVRXOBJ1");
            Encode(bytes, value);
            if (nb::isinstance<nb::dict>(value))
            {
                auto const physicalPath = fmt::format("__nvrx_native_dict_rank_{}__.{}", _scan.GetRankTopo().rank, path);
                RegisterBlob(physicalPath, std::move(bytes));
                _records.emplace_back(physicalPath, uint8_t{20});
            }
            else
            {
                RegisterBlob(path, std::move(bytes));
                _records.emplace_back(path, uint8_t{10});
            }
        }
        return nb::borrow<nb::object>(value);
    }

    PyStateDictAccessor&                         _scan;
    nb::dict                                     _types;
    std::unordered_set<PyObject*>                _active;
    std::set<std::string>                        _paths;
    std::vector<std::pair<std::string, uint8_t>> _records;
    std::unordered_set<PyObject*>                _writeActive;
    std::set<std::string>                        _writePaths;
    std::vector<WriteItem>                       _writeItems;
};
}    // namespace checkpoint::python

static constexpr std::string_view StateTypeName(StateNode::Type const& type)
{
    switch (type)
    {
    case StateNode::Type::Invalid: return "Invalid";
    case StateNode::Type::Dict: return "Dict";
    case StateNode::Type::Scalar: return "Scalar";
    case StateNode::Type::DynamicSizeBlob: return "DynamicSizeBlob";
    case StateNode::Type::DTensor: return "DTensor";
    case StateNode::Type::FixedSizeBlob: return "FixedSizeBlob";
    case StateNode::Type::Unknown: break;
    }
    return "Unknown";
}

static bool
RefreshTensorShard(DTensor& registered, TensorShard const& supplied, std::string_view path, std::string_view synchronizationState)
{
    auto previousShard = std::ranges::find_if(
        registered.shards, [&](TensorShard const& shard) { return shard.rank == supplied.rank && shard.sharddim == supplied.sharddim; });
    if (previousShard == registered.shards.end()) { return false; }
    if (previousShard->storageSize != supplied.storageSize)
    {
        throw std::runtime_error(fmt::format("State '{}' changed serialized size after registry synchronization: {} != {} bytes {}",
                                             path,
                                             supplied.storageSize,
                                             previousShard->storageSize,
                                             synchronizationState));
    }
    auto expectedShard   = *previousShard;
    expectedShard.handle = supplied.handle;
    expectedShard.device = supplied.device;
    registered.shards.erase(previousShard);
    registered.shards.insert(expectedShard);
    return true;
}
static DTensor& ShardedState(StateNode::Variant& data)
{
    if (std::holds_alternative<StateNode::FixedSizeBlob>(data)) { return std::get<StateNode::FixedSizeBlob>(data); }
    return std::get<DTensor>(data);
}

static DTensor const& ShardedState(StateNode::Variant const& data)
{
    if (std::holds_alternative<StateNode::FixedSizeBlob>(data)) { return std::get<StateNode::FixedSizeBlob>(data); }
    return std::get<DTensor>(data);
}

static bool IsShardedState(StateNode::Variant const& data)
{ return std::holds_alternative<DTensor>(data) || std::holds_alternative<StateNode::FixedSizeBlob>(data); }

namespace
{
using StateDictScanner = void (*)(PyStateDictAccessor&, nb::dict const&);

void ScanPreparedStateDict(PyStateDictAccessor& accessor, nb::dict const& stateDict)
{
    std::unordered_set<PyObject*> visited;
    RegisterStateDictRecr(accessor, StateNode::Handle::Root, stateDict, true, visited);
}

void ScanNativeStateDict(PyStateDictAccessor& accessor, nb::dict const& stateDict)
{
    nb::gil_scoped_acquire                      acquire;
    checkpoint::python::NativeStateRegistration registration(accessor);
    registration.Register(stateDict);
}

struct PythonStateTree
{
    struct Accessor : PyStateDictAccessor
    {
        Accessor(StateTree* registry, PythonStateTree* state) : _registry(registry), _state(state) {}

        [[nodiscard]] RankTopology GetRankTopo() const override { return _registry->Topo(); }
        [[nodiscard]] std::string  GetNodeId(StateNode::Handle node) const override;

        StateNode* AddDictionaryChild(StateNode::Handle parent, std::string_view name) override;
        StateNode* RegisterState(StateNode::Handle parent, std::string_view name, StateNode::Variant data) override;
        StateNode* RegisterStateByFQN(std::string_view fqn, StateNode::Variant data) override;
        void       AcquireTensorObjectRefForSnapshot(StateNode::Handle node, nb::object const& obj) override;
        void       RetainQuantHeader(std::unique_ptr<checkpoint::quant::Header> header) override;
        uint64_t   RetainBlobData(std::vector<char> data) override;

        private:
        StateTree*       _registry{nullptr};
        PythonStateTree* _state{nullptr};
    };

    explicit PythonStateTree(cipc::SessionId sessionId, RankTopology topology) : _localRegistry(sessionId, std::move(topology)) {}

    ~PythonStateTree()
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
            for (auto& object : _tensorRefObjects) { static_cast<void>(object.release()); }
            static_cast<void>(_dictToScan.release());
        }
    }

    CLASS_DELETE_COPY_AND_MOVE(PythonStateTree);

    void FinalizeRegistry()
    {
        ChainAsyncCall([this] {
            if (!_snapshotRegistry)
            {
                FinalizeStateRegistry(_localRegistry);
                _snapshotRegistry = GetSnapshotRegistry(_localRegistry);
                Scan(*_snapshotRegistry);
            }
        });
    }

    void RegisterStateDict(nb::dict stateDict, StateDictScanner scanner)
    {
        ChainAsyncCall([this, stateDict = std::move(stateDict), scanner] mutable {
            if (_scanner != nullptr && _scanner != scanner)
            {
                throw std::runtime_error("Checkpoint registration API changed after initial registration");
            }
            if (_scanner == nullptr && _localRegistry.IsFinalized())
            {
                throw std::runtime_error("Checkpoint state must be registered before registry synchronization");
            }
            _scanner = scanner;
            {
                nb::gil_scoped_acquire acquire;
                _dictToScan = stateDict;
            }
            Scan(SnapshotRegistry());
        });
    }

    void Scan(StateTree& registry);
    void SaveCheckpoint(cipc::SnapshotId iteration, std::filesystem::path const& dirpath);

    template <typename TCall> void ChainAsyncCall(TCall&& call)    // NOLINT(cppcoreguidelines-missing-std-forward)
    {
        std::scoped_lock lock(_asyncMutex);
        _async = std::async(std::launch::async,
                            [previous = std::move(_async), next = std::function<void()>(std::forward<TCall>(call))] mutable {
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

    [[nodiscard]] StateTree&       SnapshotRegistry() { return _snapshotRegistry ? *_snapshotRegistry : _localRegistry; }
    [[nodiscard]] StateTree const& SnapshotRegistry() const { return _snapshotRegistry ? *_snapshotRegistry : _localRegistry; }

    void SetStartedAt(int64_t startedAt) { SnapshotRegistry().startedAt = std::chrono::nanoseconds{startedAt}; }

    private:
    void ReleasePythonObjects_();

    void OnSnapshotComplete_()
    {
        {
            nb::gil_scoped_acquire acquire;
            ReleasePythonObjects_();
        }
    }

    RankTopology const                                      _topology;
    StateTree                                               _localRegistry;
    std::shared_ptr<StateTree>                              _snapshotRegistry;
    nb::dict                                                _dictToScan;
    std::vector<nb::object>                                 _tensorRefObjects;
    std::vector<std::unique_ptr<checkpoint::quant::Header>> _quantHeaders;
    std::vector<std::vector<char>>                          _blobStorage;
    std::mutex                                              _asyncMutex;
    std::future<void>                                       _async;
    StateDictScanner                                        _scanner{nullptr};
};
}    // namespace

StateNode* PythonStateTree::Accessor::AddDictionaryChild(StateNode::Handle parent, std::string_view name)
{
    // TODO_NO_REPARENT : We need to traverse the dictionary even if the node isnt in the snapshotTree
    if (parent == StateNode::Handle::Invalid) { return nullptr; }
    if (_registry->IsFinalized())
    {
        auto const parentPath = _registry->GetNodeId(parent);
        auto const fqn        = parentPath.empty() ? std::string(name) : fmt::format("{}.{}", parentPath, name);
        auto const handle     = _registry->FindNodeByFQN(fqn);
        // TODO_NO_REPARENT : We need to traverse the dictionary even if the node isnt in the snapshotTree
        if (handle == StateNode::Handle::Invalid) { return nullptr; }
        auto& child = _registry->GetNode(handle);
        if (!std::holds_alternative<StateNode::Dict>(child.data))
        {
            throw std::runtime_error(fmt::format("supplied state '{}' changed container type", _registry->GetNodeId(child.handle)));
        }
        return &child;
    }
    auto& child = _registry->AddChild(parent, name);
    child.data  = StateNode::Dict{};
    return &child;
}

StateNode* PythonStateTree::Accessor::RegisterStateByFQN(std::string_view fqn, StateNode::Variant data)
{
    if (_registry->IsFinalized())
    {
        auto const handle = _registry->FindNodeByFQN(fqn);
        if (handle == StateNode::Handle::Invalid) { return nullptr; }
        auto const& node = _registry->GetNode(handle);
        return RegisterState(node.parent, node.name, std::move(data));
    }
    auto [name, parent] = _registry->FindOrAddNodeByFQN(fqn);
    return RegisterState(parent, name, std::move(data));
}

StateNode* PythonStateTree::Accessor::RegisterState(StateNode::Handle parent, std::string_view stateName, StateNode::Variant data)
{
    if (parent == StateNode::Handle::Invalid) { return nullptr; }
    if (!_registry->IsFinalized())
    {
        auto [exists, childHandle] = _registry->FindChildNodeIfExists(parent, stateName);
        if (exists)
        {
            auto& child = _registry->GetNode(childHandle);
            if (IsShardedState(child.data) && data.index() == child.data.index())
            {
                auto&       registered = ShardedState(child.data);
                auto const& supplied   = ShardedState(data);
                auto const  path       = _registry->GetNodeId(child.handle);
                if (registered.durableDtype != supplied.durableDtype || registered.fulldim != supplied.fulldim)
                {
                    throw std::runtime_error(fmt::format("State '{}' changed tensor metadata before registry synchronization", path));
                }
                bool const fixedBlob = std::holds_alternative<StateNode::FixedSizeBlob>(data);
                for (auto const& shard : supplied.shards)
                {
                    if (!RefreshTensorShard(registered, shard, path, "before registry synchronization")) { registered.AddShard(shard); }
                }
                return &child;
            }
            throw std::runtime_error(fmt::format("State '{}' was registered more than once before registry synchronization "
                                                 "(existing type {}, supplied type {})",
                                                 _registry->GetNodeId(child.handle),
                                                 StateTypeName(static_cast<StateNode::Type>(child.data.index())),
                                                 StateTypeName(static_cast<StateNode::Type>(data.index()))));
        }
        auto& child = _registry->AddChild(parent, stateName);
        child.data  = std::move(data);
        return &child;
    }

    auto const parentPath  = _registry->GetNodeId(parent);
    auto const fqn         = parentPath.empty() ? std::string(stateName) : fmt::format("{}.{}", parentPath, stateName);
    auto const childHandle = _registry->FindNodeByFQN(fqn);
    if (childHandle == StateNode::Handle::Invalid) { return nullptr; }

    auto& child = _registry->GetNode(childHandle);
    if (child.data.index() != data.index())
    {
        throw std::runtime_error(fmt::format("supplied state '{}' changed registered type", _registry->GetNodeId(child.handle)));
    }
    auto const path = _registry->GetNodeId(child.handle);
    if (data.index() == static_cast<size_t>(StateNode::Type::Scalar))
    {
        auto&       previous = std::get<StateNode::Scalar>(child.data);
        auto const& current  = std::get<StateNode::Scalar>(data);
        if (current.GetType() != previous.GetType())
        {
            throw std::runtime_error(fmt::format("State '{}' changed scalar type after registry synchronization", path));
        }
        static_cast<Primitives64Bit&>(previous) = static_cast<Primitives64Bit const&>(current);
    }
    else if (data.index() == static_cast<size_t>(StateNode::Type::DynamicSizeBlob))
    {
        auto&       previous = std::get<StateNode::DynamicSizeBlob>(child.data);
        auto const& current  = std::get<StateNode::DynamicSizeBlob>(data);
        previous.handle      = current.handle;
        previous.storageSize = current.storageSize;
    }
    else if (IsShardedState(data))
    {
        auto&       previous = ShardedState(child.data);
        auto const& current  = ShardedState(data);
        if (current.durableDtype != previous.durableDtype || current.fulldim != previous.fulldim)
        {
            throw std::runtime_error(fmt::format("State '{}' changed tensor metadata after registry synchronization", path));
        }
        bool captured = false;
        for (auto const& currentShard : current.shards)
        {
            captured = RefreshTensorShard(previous, currentShard, path, "after registry synchronization") || captured;
        }
        if (!captured) { return nullptr; }
    }
    return &child;
}

void PythonStateTree::Accessor::AcquireTensorObjectRefForSnapshot(StateNode::Handle /* node */, nb::object const& obj)
{ _state->_tensorRefObjects.push_back(obj); }

void PythonStateTree::Accessor::RetainQuantHeader(std::unique_ptr<checkpoint::quant::Header> header)
{ _state->_quantHeaders.push_back(std::move(header)); }

uint64_t PythonStateTree::Accessor::RetainBlobData(std::vector<char> data)
{
    _state->_blobStorage.push_back(std::move(data));
    return reinterpret_cast<uint64_t>(_state->_blobStorage.back().data());
}

// TODO_NO_REPARENT : We need to traverse the dictionary even if the node isnt in the snapshotTree
std::string PythonStateTree::Accessor::GetNodeId(StateNode::Handle node) const
{ return node == StateNode::Handle::Invalid ? std::string{} : _registry->GetNodeId(node); }

void PythonStateTree::ReleasePythonObjects_()
{
    _tensorRefObjects.clear();
    _quantHeaders.clear();
    _blobStorage.clear();
    _dictToScan.reset();
}

void PythonStateTree::Scan(StateTree& registry)
{
    if (_scanner == nullptr) { return; }
    Accessor accessor{&registry, this};
    _scanner(accessor, _dictToScan);
}

void PythonStateTree::SaveCheckpoint(cipc::SnapshotId iteration, std::filesystem::path const& dirpath)
{

    if (!_snapshotRegistry) { throw std::runtime_error("Snapshot registry is not available"); }
    ::SaveCheckpoint(*_snapshotRegistry, iteration, dirpath);
    OnSnapshotComplete_();
}

static std::vector<char> SliceBlob(nb::handle value)
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

static std::vector<char> StringBlob(nb::handle value)
{
    auto       string = nb::cast<nb::str>(value);
    Py_ssize_t len    = 0;
    auto*      ptr    = PyUnicode_AsWideCharString(string.ptr(), &len);
    if (ptr == nullptr) { throw nb::python_error(); }
    auto const*       begin = reinterpret_cast<char const*>(ptr);    // NOLINT
    std::vector<char> blob(begin, begin + static_cast<size_t>(len * 2));
    PyMem_Free(ptr);
    return blob;
}

static std::vector<char> BytesBlob(nb::handle value)
{
    auto const bytes = nb::cast<nb::bytes>(value);
    return {bytes.c_str(), bytes.c_str() + bytes.size()};
}

template <typename T, typename TPy> static std::vector<char> TrivialTupleBlob(nb::handle value)
{
    auto              obj = nb::cast<TPy>(value);
    std::vector<char> data(sizeof(T) * obj.size());
    auto*             ptr = reinterpret_cast<T*>(data.data());                   // NOLINT
    for (size_t i = 0; i < obj.size(); ++i) { ptr[i] = nb::cast<T>(obj[i]); }    // NOLINT
    return data;
}

template <typename TPy> static std::vector<char> StringTupleBlob(nb::handle value)
{
    auto   obj  = nb::cast<TPy>(value);
    size_t size = 0;
    for (size_t i = 0; i < obj.size(); ++i) { size += (PyObject_Length(obj[i].ptr()) * sizeof(wchar_t)); }
    std::vector<char> data(size);
    auto              it = data.begin();
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

static std::vector<char> EnumListBlob(nb::list const& obj)
{
    std::vector<int64_t> values(obj.size());
    for (size_t i = 0; i < obj.size(); ++i) { values[i] = nb::cast<int64_t>(obj[i].attr("value")); }
    auto const* data = reinterpret_cast<char const*>(values.data());    // NOLINT
    return {data, data + (sizeof(int64_t) * values.size())};
}

static std::vector<char> PythonRngStateBlob(nb::tuple const& tuple)
{
    auto              state = nb::cast<nb::tuple>(tuple[1]);
    std::vector<char> data((state.size() + 1) * sizeof(int64_t));
    auto*             ptr = reinterpret_cast<int64_t*>(data.data());    // NOLINT
    ptr[0]                = nb::cast<int64_t>(tuple[0]);                // NOLINT
    for (size_t i = 0; i < state.size(); i++)
    {
        ptr[i + 1] = nb::cast<int64_t>(state[i]);    // NOLINT
    }
    return data;
}

static std::vector<char> NumPyRngStateBlob(nb::tuple const& tuple)
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
    std::vector<char> data(size);
    size_t            offset = 0;
    auto              append = [&](void const* source, size_t bytes) {
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

    if (nb::isinstance<nb::int_>(val) || nb::isinstance<nb::float_>(val) || nb::isinstance<bool>(val))
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
        // Post-`apply_factories` DTensor lists land here — register each
        // element under its own FQN before the generic per-element-type checks
        // below get a chance to mis-classify them (a list of DTensor
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
                    auto*                  child     = scan.AddDictionaryChild(node, fmt::format("{}.[list]#{}", keyView, i));
                    auto                   childDict = nb::cast<nb::dict>(item);
                    nb::gil_scoped_release release;
                    // TODO_NO_REPARENT : We need to traverse the dictionary even if the node isnt in the snapshotTree
                    RegisterStateDictRecr(
                        scan, child == nullptr ? StateNode::Handle::Invalid : child->handle, childDict, recursive, visited);
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
                        auto const elementKey = fmt::format("{}.[list]#{}.[list]#{}", keyView, i, j);
                        auto const element    = nb::borrow<nb::object>(item[j]);
                        RegisterStateDictRecrContainerItem(
                            scan, node, recursive, visited, nb::str(elementKey.c_str(), elementKey.size()), element);
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
    else if (TryRegisterTorchStateDict(scan, node, key, val)) {}
    else if (recursive)
    {
        if (nb::isinstance<nb::dict>(val))
        {
            auto*                  child     = scan.AddDictionaryChild(node, keyView);
            auto                   childDict = nb::cast<nb::dict>(val);
            nb::gil_scoped_release release;
            // TODO_NO_REPARENT : We need to traverse the dictionary even if the node isnt in the snapshotTree
            RegisterStateDictRecr(scan, child == nullptr ? StateNode::Handle::Invalid : child->handle, childDict, recursive, visited);
        }
        // if namespace
        else if (IsNamespace(val))
        {
            auto*                  child     = scan.AddDictionaryChild(node, keyView);
            auto                   childDict = nb::cast<nb::dict>(val.attr("__dict__"));
            nb::gil_scoped_release release;
            // TODO_NO_REPARENT : We need to traverse the dictionary even if the node isnt in the snapshotTree
            RegisterStateDictRecr(scan, child == nullptr ? StateNode::Handle::Invalid : child->handle, childDict, recursive, visited);
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
                RegisterStateDictRecrContainerItem(scan, node, recursive, visited, nb::str(key), value);
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

template <StateDictScanner Scanner> static void RegisterStateDict(PythonStateTree& self, nb::dict const& stateDict)
{
    auto                   extraDictRef = nb::dict(stateDict);
    nb::gil_scoped_release release;
    self.RegisterStateDict(std::move(extraDictRef), Scanner);
}

static void SaveAsync(PythonStateTree& self, int iteration, std::filesystem::path const& dirpath)
{
    nb::gil_scoped_release release;
    self.ChainAsyncCall([&self, iteration, dirpath] { self.SaveCheckpoint(static_cast<cipc::SnapshotId>(iteration), dirpath); });
}

namespace
{
using PythonTargets    = std::map<std::string, nb::object>;
using LogicalIndex     = std::tuple<std::string, bool, std::vector<uint64_t>>;
using LogicalItem      = std::pair<LogicalIndex, uint64_t>;
using LogicalPlans     = std::map<uint32_t, std::vector<LogicalItem>>;
using ObjectCandidates = std::map<uint32_t, nb::object>;

nb::object ExpectedChild(nb::handle target, nb::handle key, nb::dict const& types)
{
    if (StateIsNamespace(target, types)) { return nb::cast<nb::dict>(target.attr("__dict__")).get(key, nb::none()); }
    if (nb::isinstance<nb::dict>(target)) { return nb::cast<nb::dict>(target).get(key, nb::none()); }
    return nb::none();
}

std::string_view TakeNative(std::string_view data, size_t& cursor, size_t count)
{
    if (cursor > data.size() || count > data.size() - cursor) { throw std::invalid_argument("Truncated native checkpoint value"); }
    auto result = data.substr(cursor, count);
    cursor += count;
    return result;
}

uint64_t NativeNumber(std::string_view data, size_t& cursor, size_t width)
{
    auto     bytes = TakeNative(data, cursor, width);
    uint64_t value = 0;
    for (size_t index = 0; index < width; ++index)
    {
        value |= static_cast<uint64_t>(static_cast<unsigned char>(bytes[index])) << (index * 8);
    }
    return value;
}

std::string NativeString(std::string_view data, size_t& cursor)
{
    auto const count = NativeNumber(data, cursor, 4);
    return std::string(TakeNative(data, cursor, count));
}

std::vector<uint64_t> NativeShape(std::string_view data, size_t& cursor)
{
    auto const dimensions = NativeNumber(data, cursor, 4);
    if (dimensions > 64) { throw std::invalid_argument("Native checkpoint array has too many dimensions"); }
    std::vector<uint64_t> shape;
    shape.reserve(dimensions);
    for (uint64_t index = 0; index < dimensions; ++index) { shape.push_back(NativeNumber(data, cursor, 8)); }
    return shape;
}

nb::tuple PythonShape(std::vector<uint64_t> const& shape)
{
    nb::list result;
    for (auto const size : shape) { result.append(nb::int_(size)); }
    return nb::tuple(result);
}

nb::object NativeEnumType(nb::handle target, std::string_view path, nb::dict const& types)
{
    if (nb::isinstance(target, types["enum"])) { return nb::borrow<nb::object>(target.type()); }
    if (path == "args.exit_signal") { return nb::module_::import_("signal").attr("Signals"); }
    if (path == "args.attention_backend") { return nb::module_::import_("megatron.core.transformer.enums").attr("AttnBackend"); }
    if (path == "args.inference_cuda_graph_scope")
    {
        return nb::module_::import_("megatron.core.transformer.enums").attr("InferenceCudaGraphScope");
    }
    return nb::none();
}

nb::object DecodeNativeObject(std::string_view   data,
                              size_t&            cursor,
                              nb::handle         target,
                              std::string const& path,
                              nb::dict const&    types,
                              bool               materialize,
                              size_t             depth = 0)    // NOLINT
{
    if (depth > 100) { throw std::invalid_argument("Native checkpoint nesting exceeds the supported depth"); }
    auto const kind = NativeNumber(data, cursor, 1);
    switch (kind)
    {
    case 0: return nb::none();
    case 1: return nb::int_(std::bit_cast<int64_t>(NativeNumber(data, cursor, 8)));
    case 2: return nb::int_(NativeNumber(data, cursor, 8));
    case 3: return nb::float_(std::bit_cast<double>(NativeNumber(data, cursor, 8)));
    case 4:
    {
        auto const value = NativeNumber(data, cursor, 1);
        if (value > 1) { throw std::invalid_argument("Invalid native boolean value"); }
        return nb::bool_(value != 0);
    }
    case 5:
    {
        auto const value = NativeString(data, cursor);
        return nb::str(value.data(), value.size());
    }
    case 6:
    {
        auto const count = NativeNumber(data, cursor, 8);
        auto const value = TakeNative(data, cursor, count);
        return nb::bytes(value.data(), value.size());
    }
    case 7: [[fallthrough]];
    case 8:
    {
        auto const count = NativeNumber(data, cursor, 4);
        if (count > data.size() - cursor) { throw std::invalid_argument("Truncated native sequence"); }
        nb::list   values;
        auto const sequence = nb::isinstance<nb::list>(target) || nb::isinstance<nb::tuple>(target);
        for (uint64_t index = 0; index < count; ++index)
        {
            nb::object old = sequence && index < nb::len(target) ? nb::borrow<nb::object>(target[index]) : nb::none();
            values.append(DecodeNativeObject(data, cursor, old, fmt::format("{}.{}", path, index), types, materialize, depth + 1));
        }
        return kind == 8 ? nb::object(nb::tuple(values)) : nb::object(values);
    }
    case 9: [[fallthrough]];
    case 10:
    {
        auto const count = NativeNumber(data, cursor, 4);
        if (count > (data.size() - cursor) / 2) { throw std::invalid_argument("Truncated native dictionary"); }
        nb::dict values;
        for (uint64_t index = 0; index < count; ++index)
        {
            auto key = DecodeNativeObject(data, cursor, nb::none(), {}, types, true, depth + 1);
            if (values.contains(key)) { throw std::invalid_argument("Duplicate native dictionary key"); }
            if (kind == 10 && !nb::isinstance<nb::str>(key))
            {
                throw std::invalid_argument("Native namespace attributes require string names");
            }
            auto const known = nb::isinstance<nb::dict>(target) || StateIsNamespace(target, types);
            auto const mapping
                = known ? (StateIsNamespace(target, types) ? nb::cast<nb::dict>(target.attr("__dict__")) : nb::cast<nb::dict>(target))
                        : nb::dict();
            auto const inspect = materialize || (known && !mapping.contains(key));
            values[key]
                = DecodeNativeObject(data, cursor, ExpectedChild(target, key, types), StatePath(path, key), types, inspect, depth + 1);
        }
        if (kind == 9) { return values; }
        auto result = types["namespace"]();
        nb::cast<nb::dict>(result.attr("__dict__")).update(values);
        return result;
    }
    case 11:
    {
        auto const dtype = types["numpy"].attr("dtype")(NativeString(data, cursor));
        if (nb::cast<bool>(dtype.attr("hasobject"))) { throw std::invalid_argument("Native checkpoint arrays require a non-object dtype"); }
        auto const shape = NativeShape(data, cursor);
        auto const count = NativeNumber(data, cursor, 8);
        uint64_t   bytes = nb::cast<uint64_t>(dtype.attr("itemsize"));
        for (auto const dimension : shape)
        {
            if (dimension != 0 && bytes > std::numeric_limits<uint64_t>::max() / dimension)
            {
                throw std::overflow_error("Native checkpoint array size overflow");
            }
            bytes *= dimension;
        }
        if (bytes != count) { throw std::invalid_argument("Native array byte length does not match its shape"); }
        auto const raw = TakeNative(data, cursor, count);
        return types["numpy"]
            .attr("frombuffer")(nb::bytes(raw.data(), raw.size()), nb::arg("dtype") = dtype)
            .attr("copy")()
            .attr("reshape")(PythonShape(shape));
    }
    case 12:
    {
        auto const name  = NativeString(data, cursor);
        auto const torch = types["torch"];
        if (!nb::hasattr(torch, name.c_str())) { throw std::invalid_argument("Unknown native tensor dtype"); }
        auto dtype = torch.attr(name.c_str());
        if (!nb::isinstance(dtype, torch.attr("dtype"))) { throw std::invalid_argument("Unknown native tensor dtype"); }
        return dtype;
    }
    case 13:
    {
        auto value    = DecodeNativeObject(data, cursor, nb::none(), path, types, materialize, depth + 1);
        auto enumType = NativeEnumType(target, path, types);
        if (!enumType.is_none()) { return enumType(value); }
        if (materialize) { return value; }
        throw std::invalid_argument(fmt::format("Missing destination enum type for: {}", path));
    }
    case 14: return types["torch"].attr("device")(NativeString(data, cursor));
    case 15:
    {
        auto const name  = NativeString(data, cursor);
        auto const shape = NativeShape(data, cursor);
        bool       empty = false;
        for (auto const dimension : shape) { empty = empty || dimension == 0; }
        if (!empty) { throw std::invalid_argument("Native empty tensor has a nonempty shape"); }
        auto const torch = types["torch"];
        if (!nb::hasattr(torch, name.c_str())) { throw std::invalid_argument("Unknown native tensor dtype"); }
        return torch.attr("empty")(PythonShape(shape), nb::arg("dtype") = torch.attr(name.c_str()));
    }
    case 16:
    {
        auto const device = NativeString(data, cursor);
        auto const count  = NativeNumber(data, cursor, 8);
        auto const raw    = TakeNative(data, cursor, count);
        auto const torch  = types["torch"];
        nb::object actualDevice
            = nb::isinstance(target, types["generator"]) ? nb::object(target.attr("device")) : nb::object(nb::str(device.c_str()));
        auto result = torch.attr("Generator")(nb::arg("device") = actualDevice);
        auto state  = torch.attr("frombuffer")(nb::module_::import_("builtins").attr("bytearray")(nb::bytes(raw.data(), raw.size())),
                                               nb::arg("dtype") = torch.attr("uint8"));
        result.attr("set_state")(state);
        return result;
    }
    default: throw std::invalid_argument(fmt::format("Unsupported native checkpoint value kind: {}", kind));
    }
}

nb::object ScalarPythonValue(StateNode::Scalar const& scalar)
{
    auto const type = scalar.GetType();
    if (Primitives64Bit::Type::IsBoolean(type)) { return nb::bool_(scalar.Cast<bool>()); }
    if (Primitives64Bit::Type::IsSigned(type)) { return nb::int_(scalar.Cast<int64_t>()); }
    if (Primitives64Bit::Type::IsUnsigned(type)) { return nb::int_(scalar.Cast<uint64_t>()); }
    if (Primitives64Bit::Type::IsFloat(type)) { return nb::float_(scalar.Cast<double>()); }
    throw std::invalid_argument("Unsupported checkpoint scalar type");
}

std::string_view BlobPythonBytes(StateNode::DynamicSizeBlob const& blob)
{
    if (blob.storageSize != 0 && blob.handle == 0) { throw std::invalid_argument("Checkpoint blob has no backing storage"); }
    return {reinterpret_cast<char const*>(blob.handle), static_cast<size_t>(blob.storageSize)};    // NOLINT
}

bool SameLoadedValue(nb::handle left, nb::handle right, nb::dict const& types, size_t depth = 0)    // NOLINT
{
    if (depth > 100) { throw std::invalid_argument("Native checkpoint nesting exceeds the supported depth"); }
    if (!left.type().is(right.type())) { return false; }
    if (nb::isinstance(left, types["array"]))
    {
        auto const kind = Str(nb::str(left.attr("dtype").attr("kind")));
        return left.attr("dtype").equal(right.attr("dtype")) && left.attr("shape").equal(right.attr("shape"))
               && nb::cast<bool>(types["numpy"].attr("array_equal")(left, right, nb::arg("equal_nan") = (kind == "f" || kind == "c")));
    }
    if (nb::isinstance(left, types["tensor"]))
    {
        return left.attr("dtype").equal(right.attr("dtype")) && left.attr("shape").equal(right.attr("shape"))
               && nb::cast<bool>(types["torch"].attr("equal")(left, right));
    }
    if (nb::isinstance(left, types["generator"]))
    {
        return left.attr("device").equal(right.attr("device"))
               && nb::cast<bool>(types["torch"].attr("equal")(left.attr("get_state")(), right.attr("get_state")()));
    }
    if (nb::isinstance<nb::float_>(left) && std::isnan(nb::cast<double>(left)) && std::isnan(nb::cast<double>(right))) { return true; }
    if (StateIsNamespace(left, types)) { return SameLoadedValue(left.attr("__dict__"), right.attr("__dict__"), types, depth + 1); }
    if (nb::isinstance<nb::dict>(left))
    {
        auto const a = nb::cast<nb::dict>(left);
        auto const b = nb::cast<nb::dict>(right);
        if (a.size() != b.size()) { return false; }
        for (auto const& [key, value] : a)
        {
            if (!b.contains(key) || !SameLoadedValue(value, b[key], types, depth + 1)) { return false; }
        }
        return true;
    }
    if (nb::isinstance<nb::list>(left) || nb::isinstance<nb::tuple>(left))
    {
        if (nb::len(left) != nb::len(right)) { return false; }
        for (size_t index = 0; index < nb::len(left); ++index)
        {
            if (!SameLoadedValue(left[index], right[index], types, depth + 1)) { return false; }
        }
        return true;
    }
    return left.equal(right);
}

bool TerminalLoadedValue(nb::handle value, nb::dict const& types)
{
    if (nb::isinstance<nb::dict>(value)) { return false; }
    if (!nb::isinstance<nb::list>(value)) { return true; }
    for (auto child : nb::cast<nb::list>(value))
    {
        if (nb::isinstance(child, types["tensor"]) || !TerminalLoadedValue(child, types)) { return false; }
    }
    return true;
}

nb::object MergeLoadedSources(ObjectCandidates const&                 candidates,
                              std::string const&                      path,
                              std::map<LogicalIndex, uint32_t> const& sources,
                              nb::dict const&                         types,
                              bool                                    described,
                              size_t                                  depth = 0)    // NOLINT
{
    if (depth > 100) { throw std::invalid_argument("Native checkpoint nesting exceeds the supported depth"); }
    auto const&  first = candidates.begin()->second;
    LogicalIndex index{path, nb::isinstance(first, types["tensor"]), {}};
    if (std::get<1>(index)) { std::get<2>(index).resize(nb::len(nb::borrow<nb::object>(first.attr("shape"))), 0); }
    if (auto const source = sources.find(index); source != sources.end())
    {
        auto const selected = candidates.find(source->second);
        if (selected == candidates.end())
        {
            throw std::invalid_argument(fmt::format("Missing selected native dictionary source for: {}", path));
        }
        return selected->second;
    }
    bool dictionaries = true;
    bool lists        = true;
    for (auto const& [rank, value] : candidates)
    {
        static_cast<void>(rank);
        dictionaries = dictionaries && nb::isinstance<nb::dict>(value);
        lists        = lists && nb::isinstance<nb::list>(value) && !TerminalLoadedValue(value, types);
    }
    if (dictionaries)
    {
        nb::dict result;
        for (auto const& [rank, value] : candidates)
        {
            static_cast<void>(rank);
            for (auto const& [key, child] : nb::cast<nb::dict>(value))
            {
                static_cast<void>(child);
                if (result.contains(key)) { continue; }
                ObjectCandidates children;
                for (auto const& [owner, item] : candidates)
                {
                    auto const dictionary = nb::cast<nb::dict>(item);
                    if (dictionary.contains(key)) { children.emplace(owner, nb::borrow<nb::object>(dictionary[key])); }
                }
                result[key] = MergeLoadedSources(children, StatePath(path, key), sources, types, described, depth + 1);
            }
        }
        return result;
    }
    if (lists)
    {
        size_t count = 0;
        for (auto const& [rank, value] : candidates)
        {
            static_cast<void>(rank);
            count = std::max(count, nb::len(value));
        }
        nb::list result;
        for (size_t position = 0; position < count; ++position)
        {
            ObjectCandidates children;
            for (auto const& [rank, value] : candidates)
            {
                if (position < nb::len(value)) { children.emplace(rank, nb::borrow<nb::object>(value[position])); }
            }
            result.append(MergeLoadedSources(children, fmt::format("{}.{}", path, position), sources, types, described, depth + 1));
        }
        return result;
    }
    for (auto const& [rank, value] : candidates)
    {
        static_cast<void>(rank);
        if (!SameLoadedValue(first, value, types))
        {
            throw std::invalid_argument(
                fmt::format(fmt::runtime(described ? "Missing logical source description for conflicting dictionary value: {}"
                                                   : "Conflicting replicated dictionary value: {}"),
                            path));
        }
    }
    return first;
}

std::map<LogicalIndex, uint32_t> SelectLoadedSources(LogicalPlans const& plans)
{
    std::vector<LogicalIndex>                  order;
    std::map<LogicalIndex, std::set<uint32_t>> holders;
    std::map<LogicalIndex, uint64_t>           costs;
    std::map<uint32_t, uint64_t>               assigned;
    for (auto const& [rank, items] : plans)
    {
        assigned.emplace(rank, 0);
        for (auto const& [index, cost] : items)
        {
            auto [found, inserted] = holders.try_emplace(index);
            if (inserted) { order.push_back(index); }
            found->second.insert(rank);
            costs[index] = cost;
        }
    }
    std::map<LogicalIndex, uint32_t> selected;
    for (auto const& index : order)
    {
        auto const& ranks = holders.at(index);
        auto        rank  = *ranks.begin();
        for (auto const candidate : ranks)
        {
            if (assigned.at(candidate) < assigned.at(rank)) { rank = candidate; }
        }
        selected.emplace(index, rank);
        assigned.at(rank) += costs.at(index);
    }
    return selected;
}

void ReadNativeSchema(std::string_view data, uint32_t rank, std::map<std::string, uint8_t>& kinds, LogicalPlans& plans)
{
    size_t     cursor = 0;
    auto const magic  = TakeNative(data, cursor, 8);
    if (magic != "NVRXNST1" && magic != "NVRXNST2") { throw std::invalid_argument("Unsupported native state schema"); }
    auto const            count = NativeNumber(data, cursor, 4);
    std::set<std::string> local;
    for (uint64_t item = 0; item < count; ++item)
    {
        auto key  = NativeString(data, cursor);
        auto kind = static_cast<uint8_t>(NativeNumber(data, cursor, 1));
        if (key.empty() || !local.insert(key).second || (kind != 1 && kind != 2 && kind != 3 && kind != 4 && kind != 10 && kind != 20))
        {
            throw std::invalid_argument("Invalid native state schema entry");
        }
        auto [found, inserted] = kinds.emplace(key, kind);
        if (!inserted && found->second != kind)
        {
            throw std::invalid_argument(fmt::format("Conflicting native storage kinds for: {}", key));
        }
    }
    if (magic == "NVRXNST2")
    {
        auto&      items     = plans[rank];
        auto const itemCount = NativeNumber(data, cursor, 4);
        for (uint64_t item = 0; item < itemCount; ++item)
        {
            auto       key    = NativeString(data, cursor);
            auto const tensor = NativeNumber(data, cursor, 1);
            if (key.empty() || tensor > 1) { throw std::invalid_argument("Invalid native logical checkpoint description"); }
            auto offsets = tensor != 0 ? NativeShape(data, cursor) : std::vector<uint64_t>{};
            auto cost    = NativeNumber(data, cursor, 8);
            if (cost == 0) { throw std::invalid_argument("Native logical checkpoint costs are positive"); }
            items.emplace_back(LogicalIndex{std::move(key), tensor != 0, std::move(offsets)}, cost);
        }
    }
    if (cursor != data.size()) { throw std::invalid_argument("Trailing bytes in native checkpoint value"); }
}

std::optional<uint32_t> NativeRecordRank(std::string_view path, std::string_view prefix, std::string_view suffix)
{
    if (!path.starts_with(prefix) || !path.ends_with(suffix) || path.size() <= prefix.size() + suffix.size()) { return std::nullopt; }
    path.remove_prefix(prefix.size());
    path.remove_suffix(suffix.size());
    uint64_t value = 0;
    for (auto const digit : path)
    {
        if (digit < '0' || digit > '9') { return std::nullopt; }
        value = value * 10 + static_cast<uint64_t>(digit - '0');
        if (value > std::numeric_limits<uint32_t>::max())
        {
            throw std::invalid_argument("Native checkpoint rank exceeds its supported range");
        }
    }
    return static_cast<uint32_t>(value);
}

bool IsTorchObjectArchive(std::string_view data)
{
    constexpr std::string_view LegacyMagic = "\x80\x02\x8a\x0a\x6c\xfc\x9c\x46\xf9\x20\x6a\xa8\x50\x19\x2e";
    return data.starts_with(LegacyMagic)
           || (data.size() >= 22 && data.starts_with("PK\003\004") && data.find("/data.pkl") != std::string_view::npos
               && data.rfind("PK\005\006") != std::string_view::npos);
}

nb::object LoadArchiveBytes(std::string_view data, nb::dict const& types)
{
    auto       value   = types["torch"].attr("load")(nb::module_::import_("io").attr("BytesIO")(nb::bytes(data.data(), data.size())),
                                                     nb::arg("map_location") = "cpu",
                                                     nb::arg("weights_only") = false);
    auto const bytesIO = nb::module_::import_("io").attr("BytesIO");
    if (nb::isinstance(value, bytesIO))
    {
        value.attr("seek")(0);
        value = types["torch"].attr("load")(value, nb::arg("map_location") = "cpu", nb::arg("weights_only") = false);
    }
    return value;
}

PythonTargets DecodeLoadedObjects(StateTree const& registry, PythonTargets const& targets, nb::dict const& types, bool materialize)
{
    std::map<std::string, uint8_t> kinds;
    LogicalPlans                   plans;
    bool                           native = false;
    registry.WalkNodes([&](StateNode const& node) {
        auto const* blob = std::get_if<StateNode::DynamicSizeBlob>(&node.data);
        if (blob == nullptr) { return; }
        auto const key  = registry.GetNodeId(node.handle);
        auto const rank = NativeRecordRank(key, "__nvrx_native_state_rank_", "__");
        if (rank)
        {
            native = true;
            ReadNativeSchema(BlobPythonBytes(*blob), *rank, kinds, plans);
        }
    });
    PythonTargets                           objects;
    std::map<std::string, ObjectCandidates> dictionaries;
    registry.WalkNodes([&](StateNode const& node) {
        auto const key = registry.GetNodeId(node.handle);
        if (auto const* scalar = std::get_if<StateNode::Scalar>(&node.data))
        {
            objects.emplace(key, ScalarPythonValue(*scalar));
            return;
        }
        auto const* blob = std::get_if<StateNode::DynamicSizeBlob>(&node.data);
        if (blob == nullptr || NativeRecordRank(key, "__nvrx_native_state_rank_", "__")) { return; }
        auto const              raw        = BlobPythonBytes(*blob);
        std::string             logicalKey = key;
        std::optional<uint32_t> dictionaryRank;
        uint8_t                 kind = 0;
        if (native)
        {
            auto found = kinds.find(key);
            if (found == kinds.end()) { throw std::invalid_argument(fmt::format("Missing native storage schema for: {}", key)); }
            kind = found->second;
            if (kind == 20)
            {
                auto const split = key.find("__.");
                if (split == std::string::npos) { throw std::invalid_argument("Invalid rank-local dictionary address"); }
                dictionaryRank = NativeRecordRank(std::string_view(key).substr(0, split + 2), "__nvrx_native_dict_rank_", "__");
                if (!dictionaryRank) { throw std::invalid_argument("Invalid rank-local dictionary address"); }
                logicalKey = key.substr(split + 3);
            }
        }
        auto const expected = targets.find(logicalKey);
        if (!materialize && expected == targets.end()) { return; }
        nb::object target = expected == targets.end() ? nb::none() : expected->second;
        nb::object value;
        if (native && kind >= 1 && kind <= 4)
        {
            if (raw.size() != 8) { throw std::invalid_argument("Native scalar payloads contain exactly eight bytes"); }
            size_t     cursor = 0;
            auto const bits   = NativeNumber(raw, cursor, 8);
            if (kind == 1) { value = nb::int_(std::bit_cast<int64_t>(bits)); }
            else if (kind == 2) { value = nb::int_(bits); }
            else if (kind == 3) { value = nb::float_(std::bit_cast<double>(bits)); }
            else
            {
                if (bits > 1) { throw std::invalid_argument("Invalid native boolean value"); }
                value = nb::bool_(bits != 0);
            }
        }
        else if (native)
        {
            size_t cursor = 0;
            if (TakeNative(raw, cursor, 8) != "NVRXOBJ1") { throw std::invalid_argument("Unsupported native object payload"); }
            value = DecodeNativeObject(raw, cursor, target, logicalKey, types, materialize || expected == targets.end());
            if (cursor != raw.size()) { throw std::invalid_argument("Trailing bytes in native checkpoint value"); }
        }
        else if (IsTorchObjectArchive(raw)) { value = LoadArchiveBytes(raw, types); }
        else
        {
            value = nb::bytes(raw.data(), raw.size());
        }
        if (dictionaryRank)
        {
            if (!nb::isinstance<nb::dict>(value))
            {
                throw std::invalid_argument("Rank-local native dictionary records contain dictionaries");
            }
            dictionaries[logicalKey].emplace(*dictionaryRank, value);
        }
        else if (!objects.emplace(logicalKey, value).second)
        {
            throw std::invalid_argument(fmt::format("Duplicate native checkpoint object: {}", logicalKey));
        }
    });
    auto const sources = SelectLoadedSources(plans);
    for (auto const& [key, candidates] : dictionaries)
    {
        if (objects.contains(key)) { throw std::invalid_argument(fmt::format("Conflicting native object records: {}", key)); }
        objects.emplace(key, MergeLoadedSources(candidates, key, sources, types, !plans.empty()));
    }
    return objects;
}

nb::object
RestoreLoadedValue(nb::handle value, nb::handle target, std::string const& path, nb::dict const& types, size_t depth = 0);    // NOLINT

void RestoreLoadedMapping(nb::dict const& values, nb::dict const& target, std::string const& path, nb::dict const& types, size_t depth)
{
    auto const original = target.keys();
    for (auto const key : original)
    {
        if (!values.contains(key))
        {
            if (PyDict_DelItem(target.ptr(), key.ptr()) != 0) { throw nb::python_error(); }
        }
    }
    for (auto const& [key, value] : values)
    {
        target[key] = RestoreLoadedValue(value, target.get(key, nb::none()), StatePath(path, key), types, depth + 1);
    }
}

nb::object
RestoreLoadedValue(nb::handle value, nb::handle target, std::string const& path, nb::dict const& types, size_t depth)    // NOLINT
{
    if (depth > 100) { throw std::invalid_argument("Native checkpoint nesting exceeds the supported depth"); }
    if (nb::isinstance(target, types["enum"])) { return nb::borrow<nb::object>(target.type())(value); }
    if (nb::isinstance(value, types["generator"]))
    {
        auto result = nb::isinstance(target, types["generator"]) ? nb::borrow<nb::object>(target) : nb::borrow<nb::object>(value);
        result.attr("set_state")(value.attr("get_state")());
        return result;
    }
    if (nb::isinstance(target, types["numpy_scalar"]))
    {
        auto const kind = Str(nb::str(target.attr("dtype").attr("kind")));
        if (kind == "S" || kind == "U") { return nb::borrow<nb::object>(target.type())(value); }
        auto scalar = types["numpy"].attr("asarray")(value, nb::arg("dtype") = target.attr("dtype"));
        if (nb::cast<uint64_t>(scalar.attr("ndim")) != 0)
        {
            throw std::invalid_argument("Native value does not match the NumPy scalar destination");
        }
        return scalar[nb::tuple()];
    }
    if (StateIsNamespace(value, types))
    {
        auto result = StateIsNamespace(target, types) ? nb::borrow<nb::object>(target) : types["namespace"]();
        RestoreLoadedMapping(nb::cast<nb::dict>(value.attr("__dict__")), nb::cast<nb::dict>(result.attr("__dict__")), path, types, depth);
        return result;
    }
    if (nb::isinstance<nb::dict>(value))
    {
        auto result = nb::isinstance<nb::dict>(target) ? nb::cast<nb::dict>(target) : nb::dict();
        RestoreLoadedMapping(nb::cast<nb::dict>(value), result, path, types, depth);
        return result;
    }
    if (nb::isinstance<nb::list>(value) || nb::isinstance<nb::tuple>(value))
    {
        bool const tuple      = nb::isinstance<nb::tuple>(value);
        bool const compatible = tuple ? nb::isinstance<nb::tuple>(target) : nb::isinstance<nb::list>(target);
        nb::list   restored;
        size_t     index = 0;
        for (auto child : nb::cast<nb::sequence>(value))
        {
            auto old = compatible && index < nb::len(target) ? nb::borrow<nb::object>(target[index]) : nb::none();
            restored.append(RestoreLoadedValue(child, old, fmt::format("{}.{}", path, index), types, depth + 1));
            ++index;
        }
        if (tuple) { return StateTuple(target, restored); }
        auto result = compatible ? nb::cast<nb::list>(target) : nb::list();
        result.clear();
        result.extend(restored);
        return result;
    }
    if (nb::isinstance(value, types["array"]))
    {
        if (nb::isinstance(target, types["array"]) && target.attr("shape").equal(value.attr("shape"))
            && target.attr("dtype").equal(value.attr("dtype")))
        {
            types["numpy"].attr("copyto")(target, value);
            return nb::borrow<nb::object>(target);
        }
        return nb::borrow<nb::object>(value);
    }
    if (nb::isinstance(value, types["tensor"]))
    {
        if (nb::isinstance(target, types["tensor"]))
        {
            if (!target.attr("shape").equal(value.attr("shape")))
            {
                throw std::invalid_argument(fmt::format("Native tensor does not match the destination at: {}", path));
            }
            target.attr("detach")().attr("copy_")(value);
            return nb::borrow<nb::object>(target);
        }
        return nb::borrow<nb::object>(value);
    }
    if (nb::isinstance<nb::bytes>(value) && nb::isinstance<nb::str>(target)) { return value.attr("decode")("utf-8"); }
    return nb::borrow<nb::object>(value);
}

void FillTensorDestination(StateNode const& node, nb::handle target, nb::dict const& types)
{
    auto const& tensor = std::get<DTensor>(node.data);
    nb::object  destination
        = nb::hasattr(target, "_local_tensor") ? nb::object(target.attr("_local_tensor")) : nb::borrow<nb::object>(target);
    if (tensor.shards.empty())
    {
        if (nb::cast<uint64_t>(destination.attr("numel")()) == 0) { return; }
        throw std::invalid_argument("Loaded tensor has no local storage");
    }
    if (tensor.shards.size() != 1) { throw std::invalid_argument("Dense tensor destination requires one local shard"); }
    auto const& shard = *tensor.shards.begin();
    if (nb::cast<uint64_t>(destination.attr("numel")()) * Tensor::GetDTypeSize(tensor.durableDtype) != shard.storageSize)
    {
        throw std::invalid_argument("Loaded tensor size does not match destination storage");
    }
    if (shard.storageSize == 0) { return; }
    auto const raw   = nb::bytes(reinterpret_cast<char const*>(shard.handle), static_cast<size_t>(shard.storageSize));    // NOLINT
    auto const torch = types["torch"];
    auto       value = torch.attr("frombuffer")(nb::module_::import_("builtins").attr("bytearray")(raw),
                                                nb::arg("dtype") = TorchDtypeObject(tensor.durableDtype));
    value            = value.attr("reshape")(destination.attr("shape"));
    destination.attr("detach")().attr("copy_")(value);
}

nb::object FillLoadDestinations(StateTree const&     registry,
                                PythonTargets const& objects,
                                nb::handle           target,
                                std::string const&   path,
                                nb::dict const&      types,
                                bool                 allowPartial)
{
    std::unordered_set<PyObject*> active;
    return WalkState(target, path, types, active, [&](nb::handle target, std::string const& path) -> std::optional<nb::object> {
        if (auto found = objects.find(path); found != objects.end())
        {
            nb::object destination = nb::borrow<nb::object>(target);
            if (path == "args" && !target.is_none()) { destination = nb::module_::import_("copy").attr("deepcopy")(target); }
            return RestoreLoadedValue(found->second, destination, path, types);
        }
        if (nb::isinstance(target, types["tensor"]))
        {
            auto const handle = registry.FindNodeByFQN(path);
            if (handle != StateNode::Handle::Invalid && std::holds_alternative<DTensor>(registry.GetNode(handle).data))
            {
                FillTensorDestination(registry.GetNode(handle), target, types);
                return nb::borrow<nb::object>(target);
            }
            if (allowPartial) { return nb::borrow<nb::object>(target); }
            if (nb::hasattr(target, "device_mesh") && target.attr("device_mesh").attr("get_coordinate")().is_none())
            {
                return nb::borrow<nb::object>(target);
            }
            if (nb::cast<uint64_t>(target.attr("numel")()) == 0) { return nb::borrow<nb::object>(target); }
            throw std::invalid_argument(fmt::format("Missing checkpoint tensor: {}", path));
        }
        if (StateIsContainer(target, types)) { return std::nullopt; }
        if (allowPartial) { return nb::borrow<nb::object>(target); }
        throw std::invalid_argument(fmt::format("Missing checkpoint value: {}", path));
    });
}

RankTopology LoadTopology(uint32_t rank, uint32_t worldSize)
{
    if (worldSize == 0 || rank >= worldSize) { throw std::invalid_argument("Checkpoint rank is outside its world size"); }
    return {.rank = Rank{rank}, .worldSize = Rank{worldSize}, .ranksPerPeer = worldSize, .peersPerNode = 1, .peers = {"localhost"}};
}

void LoadPythonCheckpoint(std::filesystem::path const& path,
                          nb::dict const&              state,
                          uint32_t                     rank,
                          uint32_t                     worldSize,
                          bool                         allowPartial,
                          nb::dict const&              missingDefaults)
{
    auto                          types    = StatePythonTypes();
    auto                          topology = LoadTopology(rank, worldSize);
    StateTree                     registry(cipc::SessionId::Default, topology);
    PythonStateTree               owner(cipc::SessionId::Default, std::move(topology));
    PythonStateTree::Accessor     accessor(&registry, &owner);
    PythonTargets                 targets;
    std::unordered_set<PyObject*> active;
    WalkState(state, {}, types, active, [&](nb::handle target, std::string const& address) -> std::optional<nb::object> {
        if (!address.empty()) { targets.emplace(address, nb::borrow<nb::object>(target)); }
        if (nb::isinstance(target, types["tensor"]))
        {
            TryRegisterTorchLoadStateDict(accessor, nb::str(address.c_str(), address.size()), target);
        }
        else if (StateIsContainer(target, types)) { return std::nullopt; }
        else
        {
            accessor.RegisterStateByFQN(address, std::monostate{});
        }
        return nb::borrow<nb::object>(target);
    });
    std::vector<ByteStorage> storage;
    {
        nb::gil_scoped_release release;
        storage = checkpoint::distcp::ReadCheckpoint(path, registry, allowPartial);
    }
    auto objects = DecodeLoadedObjects(registry, targets, types, false);
    for (auto const& [key, value] : missingDefaults)
    {
        auto const address = nb::cast<std::string>(key);
        auto const target  = targets.find(address);
        if (target != targets.end() && !nb::isinstance(target->second, types["tensor"]) && !StateIsContainer(target->second, types))
        {
            objects.try_emplace(address, nb::borrow<nb::object>(value));
        }
    }
    static_cast<void>(FillLoadDestinations(registry, objects, state, {}, types, allowPartial));
}

std::optional<nb::object> FindCommonDestination(nb::handle root, std::string const& requested, nb::dict const& types)
{
    std::optional<nb::object>     result;
    std::unordered_set<PyObject*> active;
    WalkState(root, {}, types, active, [&](nb::handle value, std::string const& path) -> std::optional<nb::object> {
        if (!result && path == requested) { result = nb::borrow<nb::object>(value); }
        if (result || (!path.empty() && !requested.starts_with(path + '.')) || !StateIsContainer(value, types))
        {
            return nb::borrow<nb::object>(value);
        }
        return std::nullopt;
    });
    return result;
}

nb::dict ReadPythonCommonState(std::filesystem::path const& path)
{
    auto                     types = StatePythonTypes();
    StateTree                registry(cipc::SessionId::Default, LoadTopology(0, 1));
    std::vector<ByteStorage> storage;
    {
        nb::gil_scoped_release release;
        storage = checkpoint::distcp::ReadCheckpoint(path, registry);
    }
    auto     objects = DecodeLoadedObjects(registry, {}, types, true);
    nb::dict result;
    for (auto common = objects.begin(); common != objects.end();)
    {
        if (common->first != "common_state/shard_0_1" && !common->first.starts_with("common_state/shard_0_1.rank_"))
        {
            ++common;
            continue;
        }
        nb::object value = common->second;
        if (nb::isinstance<nb::list>(value) && nb::len(value) == 1) { value = value[0]; }
        if (!nb::isinstance<nb::dict>(value)) { throw std::invalid_argument("Expected one bundled non-tensor state dictionary"); }
        for (auto const& [key, item] : nb::cast<nb::dict>(value))
        {
            if (result.contains(key)) { throw std::invalid_argument("Duplicate bundled checkpoint object"); }
            nb::object decoded = nb::borrow<nb::object>(item);
            if (nb::isinstance<nb::bytes>(item))
            {
                auto const       bytes = nb::cast<nb::bytes>(item);
                std::string_view raw(bytes.c_str(), bytes.size());
                if (IsTorchObjectArchive(raw)) { decoded = LoadArchiveBytes(raw, types); }
            }
            result[key] = decoded;
        }
        common = objects.erase(common);
    }
    for (auto const& [key, value] : objects)
    {
        nb::str name(key.c_str(), key.size());
        auto    previous = FindCommonDestination(result, key, types);
        if (previous)
        {
            auto comparable = value;
            if (nb::isinstance(*previous, types["enum"]) || nb::isinstance(*previous, types["numpy_scalar"]))
            {
                comparable = RestoreLoadedValue(value, *previous, key, types);
            }
            if (!SameLoadedValue(*previous, comparable, types))
            {
                throw std::invalid_argument(fmt::format("Conflicting archived and independent scalar value: {}", key));
            }
        }
        else
        {
            result[name] = value;
        }
    }
    return result;
}
}    // namespace

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
    m.def("load_checkpoint",
          &LoadPythonCheckpoint,
          "Fill destinations with defaults for missing independently stored non-tensor leaves.",
          nb::arg("path"),
          nb::arg("state_dict"),
          nb::arg("rank")               = 0,
          nb::arg("world_size")         = 1,
          nb::arg("allow_partial_load") = false,
          nb::arg("missing_defaults")   = nb::dict());
    m.def("read_common_state", &ReadPythonCommonState, nb::arg("path"));
    auto cls = nb::class_<PythonStateTree>(std::move(m), "model_state");

    cls.def_static(
           "create_root",
           [](uint32_t                 sessionId,
              uint32_t                 globalRank,
              uint32_t                 globalWorldSize,
              std::vector<std::string> peers,
              uint32_t                 ranksPerPeer,
              uint32_t                 peersPerNode) {
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
               return std::make_unique<PythonStateTree>(static_cast<cipc::SessionId>(sessionId),
                                                        RankTopology{
                                                            .rank         = Rank{globalRank},
                                                            .worldSize    = Rank{globalWorldSize},
                                                            .ranksPerPeer = ranksPerPeer,
                                                            .peersPerNode = peersPerNode,
                                                            .peers        = std::move(peers),
                                                        });
           },
           nb::arg("session_id"),
           nb::arg("global_rank"),
           nb::arg("global_world_size"),
           nb::arg("peers"),
           nb::arg("ranks_per_peer") = 0,
           nb::arg("peers_per_node") = 1)
        .def_prop_ro("globalRank", [](PythonStateTree const& self) { return self.SnapshotRegistry().GlobalRank(); })
        .def_prop_ro("globalWorldSize", [](PythonStateTree const& self) { return self.SnapshotRegistry().GlobalWorldSize(); })
        .def_prop_rw(
            "startedAt",
            [](PythonStateTree const& self) { return self.SnapshotRegistry().startedAt.count(); },
            [](PythonStateTree& self, int64_t startedAt) { self.SetStartedAt(startedAt); })
        .def("register_state_dict", &RegisterStateDict<ScanPreparedStateDict>, nb::arg("state_dict"))
        .def("register_native_state_dict", &RegisterStateDict<ScanNativeStateDict>, nb::arg("state_dict"))
        .def("finalize_registry",
             [](PythonStateTree& self) {
                 nb::gil_scoped_release release;
                 self.FinalizeRegistry();
             })
        .def_prop_ro("snapshot_size",
                     [](PythonStateTree& self) {
                         nb::gil_scoped_release release;
                         self.WaitForAsyncCalls();
                         return self.SnapshotRegistry().GetTotalSnapshotStateSizeInBytes();
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
        .def("wait_for_async_completion", [](PythonStateTree& self) {
            nb::gil_scoped_release release;
            self.WaitForAsyncCalls();
        });
}
