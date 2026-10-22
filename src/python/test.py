import random  # noqa: INP001
import sys
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.parent.parent.as_posix())

import numpy as np
import nvrx_checkpointing.checkpoint
import nvrx_checkpointing.megatron.mapping
import torch
import torch.cuda

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class Color(Enum):
    RED = auto()
    GREEN = auto()
    BLUE = auto()


class TEDType(Enum):
    kFloat8E4M3 = 7  # noqa: N815
    kFloat4E2M1 = 10  # noqa: N815


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class _QuantizedTensor(torch.Tensor):
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # noqa: ANN001, ANN206
        msg = f"{cls.__name__} does not support {func}"
        raise NotImplementedError(msg)


class MXFP8Tensor(_QuantizedTensor):
    @classmethod
    def random(cls, rows: int, cols: int, device: torch.device) -> "MXFP8Tensor":
        self = torch.Tensor._make_wrapper_subclass(cls, (rows, cols), dtype=torch.bfloat16, device=device)  # noqa: SLF001
        self._fp8_dtype = TEDType.kFloat8E4M3
        self._with_gemm_swizzled_scales = False
        # E4M3 codes below 0x7F are finite; E8M0 scale 127 encodes 1.0.
        self._rowwise_data = torch.randint(0, 0x7F, (rows, cols), dtype=torch.uint8, device=device)
        scale_shape = (_round_up(rows, 128), _round_up((cols + 31) // 32, 4))
        self._rowwise_scale_inv = torch.full(scale_shape, 127, dtype=torch.uint8, device=device)
        return self


class NVFP4Tensor(_QuantizedTensor):
    @classmethod
    def random(cls, rows: int, cols: int, device: torch.device) -> "NVFP4Tensor":
        self = torch.Tensor._make_wrapper_subclass(cls, (rows, cols), dtype=torch.bfloat16, device=device)  # noqa: SLF001
        self._fp4_dtype = TEDType.kFloat4E2M1
        self._with_gemm_swizzled_scales = False
        self._nvfp4_e4m3_max = 448
        self._row_scaled_nvfp4 = False
        # E4M3 scale 0x38 encodes 1.0; an amax of 448 * 6 makes the global scale 1.0.
        self._rowwise_data = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, device=device)
        scale_shape = (_round_up(rows, 128), _round_up(cols // 16, 4))
        self._rowwise_scale_inv = torch.full(scale_shape, 0x38, dtype=torch.uint8, device=device)
        self._amax_rowwise = torch.full((1,), 448.0 * 6.0, dtype=torch.float32, device=device)
        return self


torch.manual_seed(0)

roots = {}


def workload(session_id: int, snapshot_id: int, rank: int, world_size: int) -> None:
    nvrx_checkpointing.checkpoint.ping()
    state_dict = {}
    state_dict["int"] = 42
    state_dict["double"] = 3.14
    state_dict["bool"] = True
    state_dict["string"] = "hello world"
    state_dict["slice"] = slice(1, 10, 2)
    state_dict["torch_dtype"] = torch.float32
    state_dict["enum"] = Color.GREEN

    state_dict["tuple_int"] = (1, 2, 3)
    state_dict["tuple_double"] = (1.0, 2.0, 3.0)
    state_dict["tuple_str"] = ("a", "b", "c")
    state_dict["tuple_empty"] = ()

    state_dict["list_double"] = [1.1, 2.2, 3.3]
    state_dict["list_string"] = ["a", "b", "c"]
    state_dict["list_int"] = [1, 2, 3, 4, 5]
    state_dict["list_empty"] = []
    state_dict["list_list_dict"] = [[{"a": 1}, {"b": 2}], [state_dict]]
    # state_dict["list_list_int"] = [[1, 2, 3], [4, 5, 6]]
    state_dict["list_dict"] = [
        {"list_0_double": 1.0, "list_0_bool": False},
        {"list_1_list": [9, 10, 11]},
    ]
    state_dict["dict_empty"] = {}
    state_dict["dict_mixed"] = {
        "key_int": 100,
        "key_str": "value",
        "key_tensor": torch.tensor([1, 2, 3], device=device),
    }
    state_dict["dict_self"] = {"self": state_dict}
    state_dict["tensor"] = torch.arange(0, 16, dtype=torch.int8, device=device).reshape(4, 4)
    state_dict["mxfp8_tensor"] = MXFP8Tensor.random(4, 64, device)
    state_dict["nvfp4_tensor"] = NVFP4Tensor.random(4, 64, device)
    state_dict["sharded_tensor"] = nvrx_checkpointing.megatron.mapping.ShardedTensor(
        key="sharded_tensor",
        data=torch.arange(16.0, 24.0, dtype=torch.float32, device=device).reshape(4, 2),
        dtype=torch.float32,
        local_shape=(4, 2),
        global_offset=(4 * (rank // 2), 2 * (rank % 2)),
        global_shape=(4 * ((world_size + 1) // 2), 2 * min(world_size, 2)),
        axis_fragmentations=None,
    )
    state_dict["sharded_tensor_1"] = nvrx_checkpointing.megatron.mapping.ShardedTensor(
        key="sharded_tensor",
        data=torch.arange(16.0, 24.0, dtype=torch.float16, device=device).reshape(2, 4),
        dtype=torch.float16,
        local_shape=(2, 4),
        global_offset=(rank, 0, 0),
        global_shape=(world_size, 2, 4),
        axis_fragmentations=None,
        prepend_axis_num=1,
    )

    # Multi chunk sharded tensor
    state_dict["multi_shten"] = {
        f"chunk{x}": nvrx_checkpointing.megatron.mapping.ShardedTensor(
            key="reparented.multi_shten.chunk",
            data=torch.arange(16.0, 24.0, dtype=torch.float32, device=device).reshape(1, 4, 2),
            dtype=torch.float32,
            local_shape=(1, 4, 2),
            global_offset=(x, 0, 0),
            global_shape=(11, 4, 2),
            axis_fragmentations=None,
        )
        for x in range(11)
    }
    factory_tensors = {
        "shtenfac1.int32": torch.arange(24, 32, dtype=torch.int32, device=device).reshape(2, 4),
        "shtenfac1.fp16": torch.arange(16.0, 24.0, dtype=torch.float16, device=device).reshape(2, 4),
        "shtenfac2.int32": torch.arange(24, 32, dtype=torch.int32, device=device).reshape(2, 4),
        "shtenfac2.fp16": torch.arange(16.0, 24.0, dtype=torch.float16, device=device).reshape(2, 4),
    }

    state_dict["shtenfac1"] = nvrx_checkpointing.megatron.mapping.ShardedTensorFactory(
        key="shtenfac1",
        data=factory_tensors["shtenfac1.int32"],
        build_fn=lambda *_args: [
            nvrx_checkpointing.megatron.mapping.ShardedTensor(
                key="shtenfac1.k",
                data=factory_tensors["shtenfac1.fp16"],
                dtype=torch.float16,
                local_shape=(2, 4),
                global_offset=(rank, 0, 0),
                global_shape=(world_size, 2, 4),
                axis_fragmentations=None,
                prepend_axis_num=1,
            ),
        ],
        merge_fn=None,
        # replica_id=0,
        # axis_fragmentations=None,
    )

    state_dict["shtenfac2"] = {
        "k1": nvrx_checkpointing.megatron.mapping.ShardedTensorFactory(
            key="k1",
            data=factory_tensors["shtenfac2.int32"],
            build_fn=lambda *_args: [
                nvrx_checkpointing.megatron.mapping.ShardedTensor(
                    key="shtenfac2.k1.k",
                    data=factory_tensors["shtenfac2.fp16"],
                    dtype=torch.float16,
                    local_shape=(2, 4),
                    global_offset=(rank, 0, 0),
                    global_shape=(world_size, 2, 4),
                    axis_fragmentations=None,
                    prepend_axis_num=1,
                ),
            ],
            merge_fn=None,
            # replica_id=0,
            # axis_fragmentations=None,
        ),
    }

    state_dict["sharded_object"] = nvrx_checkpointing.megatron.mapping.ShardedObject(
        key="sharded_object",
        data={"a": 1, "b": 2},
        global_offset=(0, 0),
        global_shape=(2, 2),
        # replica_id=0
    )

    state_dict["py_rng_state"] = random.getstate()
    state_dict["numpy_rng_state"] = np.random.default_rng().bit_generator.state
    # state_dict["torch_rng_state"] = torch.get_rng_state()
    # state_dict["torch_rng"] = torch.Generator()
    root_key = (session_id, rank)
    root = roots.get(root_key)
    if root is None:
        root = nvrx_checkpointing.checkpoint.model_state.create_root(
            session_id,
            rank,
            world_size,
            ["localhost"],
            ranks_per_peer=world_size,
        )
        roots[root_key] = root

    # print(state_dict["st"].__dict__)
    root.register_state_dict(state_dict)
    nvrx_checkpointing.checkpoint.ping()

    root.save_async(snapshot_id, f"metadata/session_{session_id}/iter_{snapshot_id}")
    root.wait_for_async_completion()


def run_world(session_id: int, snapshot_id: int, world_size: int) -> None:
    with ThreadPoolExecutor(max_workers=world_size) as executor:
        futures = [executor.submit(workload, session_id, snapshot_id, rank, world_size) for rank in range(world_size)]
        for future in futures:
            future.result()
    nvrx_checkpointing.checkpoint.wait_for_ckpt_finalize(session_id, snapshot_id)


try:
    run_world(1, 0, 1)
    run_world(1, 1, 1)
    run_world(2, 0, 4)
    run_world(2, 1, 4)
finally:
    nvrx_checkpointing.checkpoint.svc_spindown()

# root.register_torch_tensor("tensor", state_dict["tensor"])
# root.register_sharded_torch_tensor(
#    "sharded_tensor", state_dict["sharded_tensor"].__dict__
# )
# root.register_sharded_torch_tensor_factory(
#    "sharded_tensor_factory", state_dict["sharded_tensor_factory"].__dict__
# )
# root.register_sharded_object("sharded_object", state_dict["sharded_object"].__dict__)
