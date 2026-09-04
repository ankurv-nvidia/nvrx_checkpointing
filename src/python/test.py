import random  # noqa: INP001
import sys
import threading
from enum import Enum, auto
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.parent.parent.as_posix())

import numpy as np
import nvrx_checkpointing.checkpoint
import nvrx_checkpointing.megatron.mapping
import torch
import torch.cuda
import torch.mps

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class Color(Enum):
    RED = auto()
    GREEN = auto()
    BLUE = auto()


torch.manual_seed(0)


def workload(_session_id: int, rank: int, world_size: int) -> None:
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
    state_dict["sharded_tensor"] = nvrx_checkpointing.megatron.mapping.ShardedTensor(
        key="sharded_tensor",
        data=torch.arange(16.0, 24.0, dtype=torch.float32, device=device).reshape(4, 2),
        dtype=torch.float32,
        local_shape=(4, 2),
        global_offset=(4 * (rank // 2), 2 * (rank % 2)),
        global_shape=(8, 4),
        axis_fragmentations=None,
    )
    state_dict["sharded_tensor_1"] = nvrx_checkpointing.megatron.mapping.ShardedTensor(
        key="sharded_tensor",
        data=torch.arange(16.0, 24.0, dtype=torch.float16, device=device).reshape(2, 4),
        dtype=torch.float16,
        local_shape=(2, 4),
        global_offset=(rank, 0, 0),
        global_shape=(4, 2, 4),
        axis_fragmentations=None,
        prepend_axis_num=1,
    )

    # Multi chunk sharded tensor
    state_dict["multi_shten"] = {
        f"chunk{x}": nvrx_checkpointing.megatron.mapping.ShardedTensor(
            key="reparented.multi_shten.chunks",
            data=torch.arange(16.0, 24.0, dtype=torch.float32, device=device).reshape(1, 4, 2),
            dtype=torch.float32,
            local_shape=(1, 4, 2),
            global_offset=(x, 0, 0),
            global_shape=(11, 4, 2),
            axis_fragmentations=None,
        )
        for x in range(11)
    }

    state_dict["shtenfac1"] = nvrx_checkpointing.megatron.mapping.ShardedTensorFactory(
        key="shtenfac1",
        data=torch.arange(24, 32, dtype=torch.int32, device=device).reshape(2, 4),
        build_fn=lambda *_args: [
            nvrx_checkpointing.megatron.mapping.ShardedTensor(
                key="k",
                data=torch.arange(16.0, 24.0, dtype=torch.float16, device=device).reshape(2, 4),
                dtype=torch.float16,
                local_shape=(2, 4),
                global_offset=(rank, 0, 0),
                global_shape=(4, 2, 4),
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
            data=torch.arange(24, 32, dtype=torch.int32, device=device).reshape(2, 4),
            build_fn=lambda *_args: [
                nvrx_checkpointing.megatron.mapping.ShardedTensor(
                    key="shtenfac2.k1.k",
                    data=torch.arange(16.0, 24.0, dtype=torch.float16, device=device).reshape(2, 4),
                    dtype=torch.float16,
                    local_shape=(2, 4),
                    global_offset=(rank, 0, 0),
                    global_shape=(4, 2, 4),
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

    root = nvrx_checkpointing.checkpoint.model_state.create_root(
        rank,
        world_size,
        ranks_per_peer=world_size,
    )

    # print(state_dict["st"].__dict__)
    root.register_state_dict(state_dict)
    root.finalize_registry()
    nvrx_checkpointing.checkpoint.ping()

    root.save_async("metadata")
    root.wait_for_deferred_save()


workload(1, 0, 1)
workload(1, 0, 1)

world_size = 4
thrds = [threading.Thread(target=workload, args=(2, rank, world_size)) for rank in range(world_size)]
[t.start() for t in thrds]
[t.join() for t in thrds]

world_size = 4
thrds = [threading.Thread(target=workload, args=(2, rank, world_size)) for rank in range(world_size)]
[t.start() for t in thrds]
[t.join() for t in thrds]

# root.register_torch_tensor("tensor", state_dict["tensor"])
# root.register_sharded_torch_tensor(
#    "sharded_tensor", state_dict["sharded_tensor"].__dict__
# )
# root.register_sharded_torch_tensor_factory(
#    "sharded_tensor_factory", state_dict["sharded_tensor_factory"].__dict__
# )
# root.register_sharded_object("sharded_object", state_dict["sharded_object"].__dict__)
