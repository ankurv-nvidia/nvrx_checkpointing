#pragma once

#include "binding.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

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

    struct Scalar
    {
        StateNode::Scalar value;
        Kind              kind;
    };

    struct WriteItem
    {
        std::string           path;
        bool                  tensor;
        std::vector<uint64_t> offsets;
        uint64_t              cost;
    };

    class Visit
    {
        public:
        Visit(std::unordered_set<PyObject*>& active, nb::handle value) : _active(active), _value(value.ptr())
        {
            if (!_active.insert(_value).second) { throw std::invalid_argument("Native state contains a recursive container"); }
        }
        ~Visit() { _active.erase(_value); }
        CLASS_DELETE_COPY_AND_MOVE(Visit);

        private:
        std::unordered_set<PyObject*>& _active;
        PyObject*                      _value;
    };

    public:
    explicit NativeStateRegistration(PyStateDictAccessor& scan) :
        _scan(scan),
        _torch(nb::module_::import_("torch")),
        _tensor(_torch.attr("Tensor")),
        _dtype(_torch.attr("dtype")),
        _device(_torch.attr("device")),
        _generator(_torch.attr("Generator")),
        _mapping(nb::module_::import_("collections.abc").attr("Mapping")),
        _enum(nb::module_::import_("enum").attr("Enum")),
        _namespace(nb::module_::import_("argparse").attr("Namespace")),
        _simpleNamespace(nb::module_::import_("types").attr("SimpleNamespace")),
        _numpy(nb::module_::import_("numpy")),
        _array(_numpy.attr("ndarray")),
        _numpyScalar(_numpy.attr("generic"))
    {
        if (nb::cast<bool>(nb::module_::import_("torch.distributed").attr("is_available")()))
        {
            _dtensor        = nb::module_::import_("torch.distributed.tensor").attr("DTensor");
            _shapeAndOffset = nb::module_::import_("torch.distributed.tensor._utils").attr("compute_local_shape_and_global_offset");
        }
        else
        {
            _dtensor        = nb::none();
            _shapeAndOffset = nb::none();
        }
    }

    void Register(nb::dict const& state)
    {
        CollectMappingWriteItems({}, state, 0);
        std::vector<CheckpointWriteItem> descriptions;
        descriptions.reserve(_writeItems.size());
        for (auto const& item : _writeItems)
        {
            CheckpointWriteItem description{.fqn = item.path, .sizeBytes = item.cost, .isTensor = item.tensor};
            for (auto const offset : item.offsets)
            {
                if (description.offsets.size() == Tensor::MaxDims)
                {
                    throw std::invalid_argument("Native tensor write description exceeds the supported dimension count");
                }
                description.offsets.push_back(offset);
            }
            descriptions.push_back(std::move(description));
        }
        _scan.SetNativeWriteItems(std::move(descriptions));
        Visit guard(_active, state);
        for (auto const& [key, value] : state) { RegisterValue(Path({}, key), value); }
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

    static std::string Path(std::string_view parent, nb::handle key)
    {
        auto const name = Str(nb::str(key));
        if (name.empty()) { throw std::invalid_argument("Native state keys must have nonempty addresses"); }
        return parent.empty() ? name : fmt::format("{}.{}", parent, name);
    }

    static std::string DtypeName(nb::handle value)
    {
        auto name = Str(nb::str(value));
        if (name.starts_with("torch.")) { name.erase(0, 6); }
        return name;
    }

    bool IsNamespace(nb::handle value) const { return nb::isinstance(value, _namespace) || nb::isinstance(value, _simpleNamespace); }

    bool IsContainer(nb::handle value) const
    { return nb::isinstance<nb::dict>(value) || nb::isinstance<nb::list>(value) || nb::isinstance<nb::tuple>(value) || IsNamespace(value); }

    bool ContainsTensor(nb::handle value, size_t depth = 0)
    {
        if (depth > MaxNestingDepth) { throw std::invalid_argument("Native checkpoint nesting exceeds the supported depth"); }
        if (nb::isinstance(value, _tensor)) { return true; }
        if (!IsContainer(value)) { return false; }
        Visit guard(_active, value);
        if (nb::isinstance<nb::dict>(value) || IsNamespace(value))
        {
            auto const items = IsNamespace(value) ? nb::cast<nb::dict>(value.attr("__dict__")) : nb::cast<nb::dict>(value);
            for (auto const& [key, child] : items)
            {
                static_cast<void>(key);
                if (ContainsTensor(child, depth + 1)) { return true; }
            }
        }
        else
        {
            for (auto const child : nb::cast<nb::iterable>(value))
            {
                if (ContainsTensor(child, depth + 1)) { return true; }
            }
        }
        return false;
    }

    bool IsTerminalWriteItem(nb::handle value, size_t depth)
    {
        if (depth > MaxNestingDepth) { throw std::invalid_argument("Native checkpoint nesting exceeds the supported depth"); }
        if (nb::isinstance(value, _mapping)) { return false; }
        if (!nb::isinstance<nb::list>(value)) { return true; }
        Visit guard(_writeActive, value);
        for (auto const child : nb::cast<nb::list>(value))
        {
            if ((nb::isinstance(child, _mapping) || nb::isinstance<nb::list>(child)) && !IsTerminalWriteItem(child, depth + 1))
            {
                return false;
            }
            if (nb::isinstance(child, _tensor)) { return false; }
        }
        return true;
    }

    void CollectMappingWriteItems(std::string_view path, nb::handle value, size_t depth)
    {
        Visit      guard(_writeActive, value);
        auto const items = value.attr("items")();
        for (auto const item : nb::cast<nb::iterable>(items))
        {
            auto const pair = nb::cast<nb::tuple>(item);
            CollectWriteItems(Path(path, pair[0]), pair[1], depth);
        }
    }

    void CollectWriteItems(std::string const& path, nb::handle value, size_t depth)
    {
        if (depth > MaxNestingDepth) { throw std::invalid_argument("Native checkpoint nesting exceeds the supported depth"); }
        if (nb::isinstance(value, _mapping))
        {
            CollectMappingWriteItems(path, value, depth + 1);
            return;
        }
        if (!IsTerminalWriteItem(value, depth))
        {
            Visit  guard(_writeActive, value);
            size_t index = 0;
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
        if (!nb::isinstance(value, _tensor))
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
        if (!_dtensor.is_none() && nb::isinstance(value, _dtensor))
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
                auto const geometry = _shapeAndOffset(value.attr("shape"), value.attr("device_mesh"), value.attr("placements"));
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

    static Scalar Integer(nb::handle value)
    {
        int        overflow    = 0;
        auto const signedValue = PyLong_AsLongLongAndOverflow(value.ptr(), &overflow);
        if (PyErr_Occurred() != nullptr) { throw nb::python_error(); }
        if (overflow == 0) { return {StateNode::Scalar{static_cast<int64_t>(signedValue)}, Kind::Int}; }
        if (overflow < 0) { throw std::out_of_range("Native state integer is below the signed 64-bit range"); }
        auto const unsignedValue = PyLong_AsUnsignedLongLong(value.ptr());
        if (PyErr_Occurred() != nullptr) { throw nb::python_error(); }
        return {StateNode::Scalar{static_cast<uint64_t>(unsignedValue)}, Kind::UInt};
    }

    void Encode(Buffer& output, nb::handle value, size_t depth = 0)
    {
        if (depth > MaxNestingDepth) { throw std::invalid_argument("Native checkpoint nesting exceeds the supported depth"); }
        if (value.is_none()) { Put8(output, static_cast<uint8_t>(Kind::None)); }
        else if (nb::isinstance(value, _enum))
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
            auto const integer = Integer(value);
            Put8(output, static_cast<uint8_t>(integer.kind));
            Put64(output,
                  integer.kind == Kind::Int ? std::bit_cast<uint64_t>(integer.value.Cast<int64_t>()) : integer.value.Cast<uint64_t>());
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
            Visit guard(_active, value);
            Put8(output, static_cast<uint8_t>(nb::isinstance<nb::list>(value) ? Kind::List : Kind::Tuple));
            auto const sequence = nb::cast<nb::sequence>(value);
            Put32(output, nb::len(sequence));
            for (auto const child : sequence) { Encode(output, child, depth + 1); }
        }
        else if (nb::isinstance<nb::dict>(value) || IsNamespace(value))
        {
            Visit      guard(_active, value);
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
        else if (nb::isinstance(value, _array))
        {
            nb::object const dtype = value.attr("dtype");
            if (nb::cast<bool>(dtype.attr("hasobject")) || !dtype.attr("fields").is_none())
            {
                throw std::invalid_argument("Native state supports NumPy arrays with primitive dtypes");
            }
            auto const array = _numpy.attr("ascontiguousarray")(value);
            Put8(output, static_cast<uint8_t>(Kind::Array));
            PutString(output, Str(nb::str(dtype)));
            PutShape(output, value.attr("shape"));
            PutByteString(output, nb::cast<nb::bytes>(array.attr("tobytes")()));
        }
        else if (nb::isinstance(value, _numpyScalar))
        {
            auto const item = value.attr("item")();
            if (nb::isinstance(item, _numpyScalar))
            {
                throw std::invalid_argument("Unsupported native checkpoint value type: NumPy scalar without a Python equivalent");
            }
            Encode(output, item, depth);
        }
        else if (nb::isinstance(value, _dtype))
        {
            Put8(output, static_cast<uint8_t>(Kind::Dtype));
            PutString(output, DtypeName(value));
        }
        else if (nb::isinstance(value, _device))
        {
            Put8(output, static_cast<uint8_t>(Kind::Device));
            PutString(output, Str(nb::str(value)));
        }
        else if (nb::isinstance(value, _generator))
        {
            nb::object const device = value.attr("device");
            Put8(output, static_cast<uint8_t>(Kind::Generator));
            PutString(output, Str(nb::str(device)));
            auto const bytes = value.attr("get_state")().attr("cpu")().attr("numpy")().attr("tobytes")();
            PutByteString(output, nb::cast<nb::bytes>(bytes));
        }
        else if (nb::isinstance(value, _tensor) && nb::cast<uint64_t>(value.attr("numel")()) == 0)
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

    void RegisterValue(std::string const& path, nb::handle value)
    {
        if (nb::isinstance(value, _tensor))
        {
            ReservePath(path);
            if ((!_dtensor.is_none() && nb::isinstance(value, _dtensor)) || nb::cast<uint64_t>(value.attr("numel")()) != 0)
            {
                if (!TryRegisterTorchStateDict(_scan, StateNode::Handle::Root, nb::str(path.c_str(), path.size()), value))
                {
                    throw std::invalid_argument(fmt::format("Native tensor '{}' cannot be registered", path));
                }
                return;
            }
        }
        else if (IsContainer(value) && ContainsTensor(value))
        {
            Visit guard(_active, value);
            if (nb::isinstance<nb::dict>(value) || IsNamespace(value))
            {
                auto const mapping = IsNamespace(value) ? nb::cast<nb::dict>(value.attr("__dict__")) : nb::cast<nb::dict>(value);
                for (auto const& [key, child] : mapping) { RegisterValue(Path(path, key), child); }
            }
            else
            {
                size_t index = 0;
                for (auto const child : nb::cast<nb::iterable>(value))
                {
                    RegisterValue(fmt::format("{}.{}", path, index), child);
                    ++index;
                }
            }
            return;
        }
        else
        {
            ReservePath(path);
        }

        if (!nb::isinstance(value, _enum) && nb::isinstance<bool>(value))
        {
            _scan.RegisterStateByFQN(path, StateNode::Scalar{static_cast<uint64_t>(nb::cast<bool>(value))});
            _records.emplace_back(path, uint8_t{4});
        }
        else if (!nb::isinstance(value, _enum) && nb::isinstance<nb::int_>(value))
        {
            auto integer = Integer(value);
            _scan.RegisterStateByFQN(path, std::move(integer.value));
            _records.emplace_back(path, static_cast<uint8_t>(integer.kind));
        }
        else if (!nb::isinstance(value, _enum) && nb::isinstance<nb::float_>(value))
        {
            _scan.RegisterStateByFQN(path, StateNode::Scalar{nb::cast<double>(value)});
            _records.emplace_back(path, uint8_t{3});
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
    }

    PyStateDictAccessor&                         _scan;
    nb::module_                                  _torch;
    nb::object                                   _tensor;
    nb::object                                   _dtype;
    nb::object                                   _device;
    nb::object                                   _generator;
    nb::object                                   _mapping;
    nb::object                                   _enum;
    nb::object                                   _namespace;
    nb::object                                   _simpleNamespace;
    nb::module_                                  _numpy;
    nb::object                                   _array;
    nb::object                                   _numpyScalar;
    nb::object                                   _dtensor;
    nb::object                                   _shapeAndOffset;
    std::unordered_set<PyObject*>                _active;
    std::set<std::string>                        _paths;
    std::vector<std::pair<std::string, uint8_t>> _records;
    std::unordered_set<PyObject*>                _writeActive;
    std::set<std::string>                        _writePaths;
    std::vector<WriteItem>                       _writeItems;
};
}    // namespace checkpoint::python
