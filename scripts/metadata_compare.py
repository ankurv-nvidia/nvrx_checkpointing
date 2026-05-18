import re
from pathlib import Path
import pickle
import torch
from torch.distributed.checkpoint import TensorStorageMetadata, ChunkStorageMetadata

checkpoint_dir = "/lustre/fsw/portfolios/coreai/users/chengsongz/megatron-lm/runs/checkpoints/phase3_rerun/iter_0000001"
metadata = pickle.load(open(f"{checkpoint_dir}/.metadata", "rb"))
common = torch.load(f"{checkpoint_dir}/common.pt", map_location="cpu")

for path_str, meta_obj in metadata.state_dict_metadata.items():
    if not isinstance(meta_obj, TensorStorageMetadata):
        continue
    print(f"{path_str}: {meta_obj}")

root = {}
for path_str, meta_obj in metadata.state_dict_metadata.items():
    if not isinstance(meta_obj, TensorStorageMetadata):
        continue # byte storage metadata is currently removed
    # TODO: chained optimizer metadata is missing
    if ".optimizer." in path_str:
        continue
    # if we see experts.experts., we edit to experts. this is due to a bug in megatron
    if "experts.experts." in path_str:
        path_str = path_str.replace("experts.experts.", "experts.")
    root[f"{path_str}.dtype"] = meta_obj.properties.dtype
    root[f"{path_str}.layout"] = meta_obj.properties.layout
    root[f"{path_str}.requires_grad"] = meta_obj.properties.requires_grad
    root[f"{path_str}.memory_format"] = meta_obj.properties.memory_format
    root[f"{path_str}.pin_memory"] = meta_obj.properties.pin_memory
    root[f"{path_str}.shape"] = meta_obj.size
    # Flatten the value here
    chunks = []
    for i, chunk in enumerate(meta_obj.chunks):
        root[f"{path_str}.chunks.{i}.offsets"] = chunk.offsets
        root[f"{path_str}.chunks.{i}.sizes"] = chunk.sizes

# common["args"] is Namespace. Save all attributes of args into root
for attr in dir(common["args"]):
    if not attr.startswith("_"):
        # if attr is a list, save it as .0, .1, .2, ...
        if isinstance(getattr(common["args"], attr), list):
            for i, value in enumerate(getattr(common["args"], attr)):
                root[f"{attr}.{i}"] = value
        else:
            root[f"{attr}"] = getattr(common["args"], attr)

# some value of common[keys] is a dict. Save all items in model into root
for key in ["opt_param_scheduler", "content_metadata"]:
    for key_next in common[key].keys():
        root[f"{key_next}"] = common[key][key_next]

for idx in common["optimizer"].keys():
    root[f"{idx}.param_state_sharding_type"] = common["optimizer"][idx]["param_state_sharding_type"]

# for all other keys in common, save them as is
for key in common.keys():
    if key not in ["state_dict_metadata", "optimizer", "model", "args", "content_metadata", "opt_param_scheduler"]:
        root[f"{key}"] = common[key]

with (Path(checkpoint_dir) / f"megatron_python_metadata.txt").open("w") as _f:
        for k, v in root.items():
            _f.write(f"{k}: {v}\n")

megatron_python_metadata_path = f"{checkpoint_dir}/megatron_python_metadata.txt"
# load file as a str: str dict. file is txt as str: str format
megatron_python_metadata = {}
with open(megatron_python_metadata_path, "r") as _f:
    for line in _f:
        try:
            key, value = line.strip().split(": ")
            megatron_python_metadata[key] = value
        except:
            print(line)


def parse_torch_string(value_str):
    """Helper to parse the string representations of torch objects from the text."""
    if 'torch.Size' in value_str:
        # Extract numbers from "torch.Size([32, 1344, ...])"
        dims = re.findall(r'\d+', value_str)
        return [int(d) for d in dims]
    if 'torch.bfloat16' in value_str:
        return 'BFloat16'
    if 'torch.float16' in value_str:
        return 'Float16'
    if 'torch.float32' in value_str:
        return 'Float32'
    if value_str == 'False':
        return '0'
    if value_str == 'True':
        return '1'
    return value_str

def convert_megatron_to_user_format(megatron_data, num_processes=2):
    """
    Converts Megatron chunks into per-process user format dicts.
    """
    # 1. Parse the flat text data into a structured object
    base_info = {}
    chunks = {}
    other_info = {}
    
    # We identify the base parameter name (e.g. decoder...linear_fc1.weight)
    # by stripping .chunks.X... or .shape, .dtype etc.
    
    for key, value in megatron_data.items():
        parsed_value = parse_torch_string(value)
        if ".layers." not in key and ".weight." not in key:
            other_info[key] = parsed_value
            continue
        
        if ".chunks." in key:
            # key example: decoder...weight.chunks.0.offsets
            pattern = r"(.*)\.chunks\.(\d+)\.(.*)"
            match = re.search(pattern, key)
            if match:
                base_name, chunk_idx, prop = match.groups()
                chunk_idx = int(chunk_idx)
                
                if base_name not in chunks:
                    chunks[base_name] = {}
                if chunk_idx not in chunks[base_name]:
                    chunks[base_name][chunk_idx] = {}
                
                chunks[base_name][chunk_idx][prop] = parsed_value
        else:
            # Global properties like shape, dtype
            # Remove the suffix to get base name
            base_name = key.rsplit('.', 1)[0]
            prop = key.rsplit('.', 1)[1]
            
            if base_name not in base_info:
                base_info[base_name] = {}
            base_info[base_name][prop] = parsed_value

    # 2. Distribute chunks to processes
    process_dicts = [other_info.copy() for _ in range(num_processes)]
    
    for base_name, chunk_dict in chunks.items():
        sorted_indices = sorted(chunk_dict.keys())
        total_chunks = len(sorted_indices)
        chunks_per_proc = total_chunks // num_processes
        
        global_shape = base_info[base_name].get('shape')
        dtype = base_info[base_name].get('dtype')

        for proc_id in range(num_processes):
            # Determine which chunks belong to this process
            if chunks_per_proc == 0:
                start_idx, end_idx = 0, total_chunks
            else:
                start_idx = proc_id * chunks_per_proc
                end_idx = start_idx + chunks_per_proc
            
            # The user format resets the weight index (weight0...weightN) locally
            local_counter = 0
            
            for global_chunk_idx in range(start_idx, end_idx):
                chunk_data = chunk_dict[global_chunk_idx]
                
                # Construct the new key prefix: e.g. decoder...linear_fc1.weight0
                # Note: 'weight' is replaced by 'weightX' in your format logic
                if end_idx - start_idx > 1:
                    prefix = base_name.replace('.weight', f'.weight{local_counter}')
                else:
                    prefix = base_name
                
                # Calculate Shard Dim: [(offset, offset+size), ...]
                offsets = chunk_data['offsets']
                sizes = chunk_data['sizes']
                shard_dim = []
                for o, s in zip(offsets, sizes):
                    shard_dim.append((o, o + s))
                
                # Write to process dict
                p_dict = process_dicts[proc_id]
                p_dict[f"{prefix}.sharddim"] = str(shard_dim)
                p_dict[f"{prefix}.fulldim"] = str(global_shape)
                p_dict[f"{prefix}.dtype"] = dtype
                p_dict[f"{prefix}.device"] = "cuda" # Hardcoded based on sample
                
                local_counter += 1

    return process_dicts

metadata_converted = convert_megatron_to_user_format(megatron_python_metadata)

cpp_metadata_path_0 = f"{checkpoint_dir}/cpp_metadata_0.txt"
cpp_metadata_path_1 = f"{checkpoint_dir}/cpp_metadata_1.txt"
# load file as a str: str dict. file is txt as str: str format
cpp_metadata_0 = {}
cpp_metadata_1 = {}
with open(cpp_metadata_path_0, "r") as _f:
    for line in _f:
        try:
            key, value = line.strip().split(": ")
            cpp_metadata_0[key] = value
        except:
            print(line)
with open(cpp_metadata_path_1, "r") as _f:
    for line in _f:
        try:
            key, value = line.strip().split(": ")
            cpp_metadata_1[key] = value
        except:
            print(line)

# get keys that only appear in cpp_metadata and only in metadata_converted
only_in_cpp = set(cpp_metadata_0.keys()) - set(metadata_converted[0].keys())
only_in_converted = set(metadata_converted[0].keys()) - set(cpp_metadata_0.keys())

print("tensor metadata only in cpp: ")
only_in_cpp_keys = []
only_in_cpp_metadata = []
for key in only_in_cpp:
    if "decoder." in key or ".layers." in key or ".weight." in key:# or "dtype_param_torch." in key:
        only_in_cpp_keys.append(key)
    else:
        only_in_cpp_metadata.append(key)
only_in_cpp_keys.sort()
for key in only_in_cpp_keys:
    print(key)

print("other metadata only in cpp: ")
only_in_cpp_metadata.sort()
for key in only_in_cpp_metadata:
    print(key)

print("tensor metadata only in megatron python: ")
only_in_converted_others = []
for key in only_in_converted:
    if "decoder." in key or ".layers." in key or ".weight." in key or "dtype_param_torch." in key:
        print(key)
    else:
        only_in_converted_others.append(key)

only_in_converted_others.sort()
print("\nother metadata only in megatron python: ")
for key in only_in_converted_others:
    if metadata_converted[0][key] is not None:
        print(key)

# verify if common keys are the same
keys_in_both = set(cpp_metadata_0.keys()) & set(metadata_converted[0].keys())
for key in keys_in_both:
    if cpp_metadata_0[key] != metadata_converted[0][key]:
        print(f"{key}: {cpp_metadata_0[key]} != {metadata_converted[0][key]}")

# save metadata_converted as 2 txt key: val files
with open(f"{checkpoint_dir}/megatron_python_metadata_converted_0.txt", "w") as _f:
    for key, value in metadata_converted[0].items():
        _f.write(f"{key}: {value}\n")
with open(f"{checkpoint_dir}/megatron_python_metadata_converted_1.txt", "w") as _f:
    for key, value in metadata_converted[1].items():
        _f.write(f"{key}: {value}\n")
