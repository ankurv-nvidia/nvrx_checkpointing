# Design Goals

- Scalability @100K+: Predictable and consistent performance

- Speed-of-Light Recovery
  - Enable the fastest path of recovery for happy-path scenarios without sacrificing recovery for other types of failures (no trade-offs)
    For eg:
    - Recovery can be catagorized into:
      - Transient: No HW resource swaps needed
      - Catastrophic : Requires recovering with a new set of GPU/Host nodes
    - Transient/Catastrophic jobs can be further categorized into
      - Low resource utilization
      - High resource utilization
    - The goal involves either auto-detecting and provding hooks for the user to select the appropriate mechanism that provides the SOL path given the resource constraints.
  - Minimizing and efficient utilization of resources
  - Fine-Grained synchronization support

- Workload Generality
  - Complex optimizers (scaling state-dictionary/sharding-metadata size and complexity)
  - Multiple tenant and models (Post-training)
  - Inference workload applicability
    - Disaggregated Prefill/decode
    - Avoid strong pytorch dependency
  - Auto-Quantization/Dequantization for checkpointing

- Infrastructure generality, redundancy and resiliency
  - Tunable policies for Mem-IO / Network-IO / Disk-IO preferences
    - There are cases where network-io maybe faster than disk-io
    - Disk might not be big enough
    - Not enough RAM
  - Redundancy by using multiple storage policies ()
  - Multiple network backends for data interchange (NVLink/Infiniband/TCP)

# API Surface

The sequencing of checkpointing and the API surface is divided into 3 main areas

1. State-Dictionary Registration
2. Save: Snapshot capture (Tensor, Blobs, Scalars etc) and durability guarantees
3. Load: Recover from snapshots

## State Registration

### Preface: Pytorch/Megatron-LM : State-Dictionary

Pytorch and megatron apis follow the model of preparing state-dictionary completely outside the visibility of the checkpointing module and have a single api to hand over that state-dictinary.

This can often lead to the checkpointing module surfacing errors that could be traced back to the invalid or erroneous state-dictionary preparation.

- State dictionary construction is very janky. No schema
- Optimizer frequently rely on trampling over child-optimizer states thus causing confusion
- State dictionary goes through multiple rounds of filtering and transformations in checkpointing code leading to unclear contract between Modules and checkpointing

### Design Goals

- Allow for "fill-in" state-dictionary preparation rather than a "hand-over"
- easy to use api-wrappers for python (megatron-lm, pytorch) based state-dictionary registration for smoother integration.
  - Support pytorch/megatron-lm bridges for DTensor, ShardedTensor metadata extraction.
- Flexibility for future (possibly inference) based usage to populate a state-tree in-place directly from c++ or other frontend languages and register the supported types.
  - Type-safe and faster
  - Better construction and more meaninful errors around collisions, duplications etc
- Future work: Capture Full fidelity tensor layout (sharding + replication) information.

```python
# First Registration
# Happens after the load-checkpoint call
root = nvrx_checkpointing.checkpoint.model_state.create_root()
root.register_state_dict(state_dict)
# Intended to signify that the state-dictionary structure wont change
root.finalize_registry()
# Finalize also warms up the SHMem-Queue and initiates a sync across all the ranks to infer the dedup-ing of replicated tensors.

...

# Update State-dictionary

# Happens as part of save_checkpoint
# Captures the blobs and update the scalar values synchronously
# Tensors are captured asynchronously
root.register_state_dict(state_dict)
```

## Save Checkpoint: Megatron

```python
# args compatible with save_checkpoint fn
def _save_checkpoint_impl(save_args: SaveCheckpointArgs, started_at_ns: int) -> None:
  # Wait for previous checkpoint to complete
    maybe_finalize_async_save(blocking=True)

  # update rng-state-blobs and hyper-parameters in state-dict

  # register state-dictionary and capture blobs + scalars
  root = register_state_dict(state_dict, iteration)

  # async scalar, blob capture + async snapshot (tensor) capture and durability
  root.save_async(iteration, checkpoint_dir)

  # wait for scalar, blob capture to complete. Tensor snapshot continues ...
  root.wait_for_deferred_save()

  # For sync-checkpointing. Wait for everything to complete
  if not getattr(args, "async_save", False):
      checkpoint.wait_for_ckpt_finalize(_SESSION_ID, iteration)
      finalize_fn()

  # For async checkpointing. oncomplete finalize-fn
  else:
      future = _get_executor().submit(checkpoint.wait_for_ckpt_finalize, _SESSION_ID, iteration)
      _pending = _PendingSave(future, finalize_fn)

```

## Load Checkpoint (TBD)

## Future API Changes

- python async/await pattern for async apis
  - fine-grained async futures for D2H, Mem-Snapshot, Durable-Snapshot etc
- Async Load-Checkpoint
  - Allow construction of data-loaders, cuda-graphs etc in parallel

## TODO : Cleanups

- namespaces
- packaging
- daemon vs session : note on why sessions are needed

- Bugfix : for session overlap

## TODO : Assumptions

- Daemon relies on megatron to validate the checkpoint file
- Shared - memory - Daemon
  - Add a Time sequence diagram for daemon and shared-memory

1. Daemon opens the shm file (listener)
   1.1 . shm file is already exists. it'll reinitialize it
   1.2 - Any producer thats still writing to the old file will error

# Resiliency and Error Handling

There are 2 main high level design philosophies that important to remember when thinking about resiliency and error propagation.

1. The daemon never intentionally crashes or raise exceptions
2. Almost all of the client exposed api surface is async.

A training job will usually go through these high level steps
(Happens once per training cycle)

1. Daemon boot (async)
2. State-Dict registration (async)
3. Save-planning globally across all ranks (background)
   (Repeated multiple times)
4. Snapshot activation
   1. Sync wait for save-planning
   2. Sync wait for last-snapshot completion
5. Snapshot transfer to daemon
6. Snapshot durability to disk
7. Snapshot disk finalization
8. Client notification and finalization callbacks
   1. GIL lock for client routine callbacks

There are several other (fatal) corner cases that will cause a fail-fast of the training process or the daemon but the primary outlets for most errors will have two main outlets for error propagation.

- Step-4.1 : Snapshot activation
  - Sync wait for save-planing covers all errors from Step-1-2-3 and bubbles any errors
- Step-8.1 : Finalization Callbacks
  - GIL Lock for client routine callbacks used for bubbling up errors between Step-4-8

In each of these cases the daemon does its best to transmit and broadcast errors to all clients as needed but sometimes the errors will only be limited to the rank it originates from.

## TODO : Error scenario examples

### Example 1 : Transient Node Failure

### Example 2 : Permanent Node Failure

### Example 3 : Session overlap

# Lifecycle Management (Daemon)

The section covers mainly the daemon-module where the main processing of state-dictionary, planning and checkpointing tasks are happening.

The daemon can operate in 2 life-cycle modes

1. Self-managed
2. Externally launched and managed

The module itself can be instantiated in 2 ways.

1. In-Proc mode (always self-managed)
2. Out-of-proc mode: Both Self-managed and external managed.

## Self-Managed

The primary mechanism to assist the self-managed lifecyle is via <mark>(TODO: better name)references</mark>. which come in 3 flavors

1. Self-References / In-Proc references
2. Local-Process Reference
3. Remote Process Reference.

These references keep the daemon process alive and assist with the startup sequence.
The daemon is auto-launched when the first-reference is acquired and gracefully spins down when the last-reference is released (after a configurable inactivity timeout)

### Startup Sequence

- The first reference in a process pings the server.
- A boot is requested if its unable to ping the server
- Potentially multiple requests can be made to bootup the server from multiple threads
- Each startup request tries to listen to the port which acts as a gate.
- The first one wins. The rest gracefully cede.
- A reference is acquired on the newly started server / already-existing server which keeps it alive until its released

### Spin-Down

- The daemon keeps a list of all references.
- As long as atleast one in-proc reference is alive it does nothing.
- When all in-proc references are released it checks the local-proc references
- Local-proc references

### Keep-Alive

## In-proc mode (No ft-launcher integration needed)

The in-proc mode implies that the daemon will be instantiated inside one of the N-local ranks in the training process and its lifetime is bound by the training process and hence the lifecycle is always self-managed.

This is meant to simplify the lifecycle management for scenarios that dont require fault-tolerance or quick recovery (mainly dev scenarios) and hence it requires no additional management.

The daemon holds the rank-process its instantiated in, from exiting, especially if its a coordinator daemon for the job and keeps the process alive unless its ready to wind-down in case of a regular spin-down.
This is part of the regular

## Out-of-proc mode : FT-Launcher Integration (Fault-Tolerance)

The ft-launcher is component responsible for tracking the individual rank processes and restarting in case of errors or exceptions happening on any of the ranks.
The section describes extension of the ft-launcher's umbrella to include tracking and crash-diagnostics on the daemon.
While the daemon does provide auto-start/spin-down behavior to avoid requiring a separate pre-launch mechanism the ft-launcher integration forces manual lifetime management by launching it manually (epilogue to training jobs) as well as spinning down manually (prologue)

In an ideal world we dont expect the daemon to ever crash or exit during a training run so all unexpected process terminations are considered fatal and invoke the fault-diagnostics as well as fault-recovery routines, causing the training to restart.

Here's a consolidated list of behaviors when it comes to ft-launcher and daemon interaction

- Cold start: Daemon to be launched by ft-launcher before training-jobs-processes
- Recovery start : Daemon is launched if not already present
  - The previous daemon process from earlier is continued to be used across all nodes.
  - The cycle id is used as a session-id for the daemon which insulate the workflow of this cycle from any partial workflow state from last cycle
  - This mechanism in future will be used to provide quick-load-checkpoint by using the in-memory snapshot from previous session.

- Logging: Redirect to itself.
  - ft-launcher launches daemon with cli-args (<mark>TODO</mark>)to capture its logging
  - One consolidated log per cycle
  - On a restart if the daemon is reused ft-launcher handles redirecting the logs to a new cycle without the need to change daemon log outputs

- Daemon Critical Fault Handling : Collect diagnostic logs and restart
  - All Daemon crashes are considered unexpected bugs that need fixing or handling.
  - Non-crashed daemons are reused. (New session-id insulates the session from last-session state)
    - <mark>TODO</mark> : Should we terminate all daemons ??. This will handicap fast-recovery
  - Diagnostics include daemon coredump (if applicable) and logs.
  - Also includes daemon being unresponsive or hanged (stack trace if hanged)

# Key Components and Code Walkthrough

### In-proc: State-Extraction

- Files
  - checkpoint.h/cpp
  - python/bindings.h/cpp
  - python/torch_bindings.cpp
  - state.h
  - stateserdes.h
  - serdes_utils.h

#### State-Tree

- state.h
  - The in-memory state representation for all the scalars/blobs/tensorshard extracted from state-dictionary
  - Blob + Scalars : Snapshotted during traversal (Includes, RNG State, learning-rate-hyperparametets etc. )
  - TensorShards : Only metadata extracted. D2H Happens asynchronously

#### Python-Interfaces

- python/bindings.h/cpp
  - Python apis registration
  - Python state dictionary traversal. (Periodic quick GIL acquisition and release to avoid stalls)
  - Extract state from state-dictionary (python) into native in-memory tree.
- python/torch_bindings.cpp
  - metadata bridge for pytorch/megatron-lm classes (ShardedTensor/ShardedObjects etc) (TODO: DTensor support)

#### D2H (and TODO: H2D)

- Device Interfacing (non-blocking)
  - Background: Tensor data extraction
    - Temporary (CPU-Mem) Bounce Buffers (`4 * 16Mb` seems sufficient) cap the total CPU memory requirements
    - No additional GPU memory needed
  - Stream/Events for Workload synchronization
- Interfacing with Checkpointing Service
  - Shared memory + sockets

### Snapshot Daemon

- Intended to persist beyond a job (across requeues)
- Multiple simultaneous job/rank sessions. (Separate shmem channel)

- Auto-Start Auto-stop Daemon Service
  - Spin down based on configured inactivity duration
  - Spin up on-demand access from in-proc client thread

- Persistent storage of checkpoints policies
  - In-memory cache (LRU) (No deduplication of replicated/partially sharded tensors)
  - Local-Disk (No deduplication)
  - Network storage (with deduplication)

- Mesh broadcasts and signalling (over separate process group disjoint from workload)
  - Broadcast metadata (local tensor shards)
  - Rendezvous for determining successful checkpoint task completion.

- Allow blocking or non-blocking persistent serialization (local-disk/network-disk) based on workload intent

- Can be configured as a part of nvrx launcher / system-service as needed
  - Perhaps we should just fuse it with the nvrx launcher ?!!

# Workflow: Out-of-proc Checkpointing-Server (Mesh)

Considering a scenario where a workload (typically a training job) consists of some fixed set of GPUs (4 in the diagram below).
Each node will have a singleton process which operates as a mesh-node, part of an overall distributed server mesh.

The node characteristics:

- Low-latency shared memory (numa aware) based data-sharing between Server and rank processes
- Multiple Backing storage
  - In-memory cache
  - Local Disk
  - Network file-system (lustre)
- Async Signalling between Mesh nodes to determine the fully backed up checkpoints.
  - Rendezvous and stall points for guarantees as needed
  - Async event notifications (registered callbacks) to the workload as needed
- Metadata broadcast between mesh nodes for identifying local and remote shards and global locality and replication.

![Failure](./Checkpoint-failure.drawio.png)

Recovery for the optimal scenario.

The following diagram illustrates a typical fast recovery where a GPU on Node-256 fails but the node itself is alive with its in-memory checkpoint available.
In this case no other model changes to parallelization schemes were made and the job is simply recovering with a hot-spare (Node-2048)

- Ranks[0-1023] have their local checkpoints valid and available in the in-memory cache for fully local and instant recovery.
- The ranks[Prev: 1028-8191] are relocated and left-shifted from the failure index with the devices for Ranks[Prev: 1024-1027] are yanked out.
  - The local checkpointed data contained in the server node (257) (Prev-Rank: 1028, New-Rank: 1024) now corresponds to +4 rank (Prev-Rank: 1032, New-Rank: 1028)
  - The Node(258) (New-Rank:[1028-1031]) pulls its data from Node-257 (New-Rank:[1024-1027]) over the networking layer thereby avoiding the disk
  - Depending on performance characteristics of the remote location (within-rack / across-rack) the mesh-node might decide to either pull its data from a remote-node or a remote-file-system
- The mesh nodes will purge their local-data (LRU cache) as needed with new metadata and local-data checkpoints.

![Failure](./Checkpoint-recovery.drawio.png)

## Workflow: Time sequence

```mermaid
sequenceDiagram

box Rank-1
participant native1
end
box Rank-0
    participant gpu0
    participant python0
    participant native0
end
participant Daemon0

participant Daemon1
participant DaemonN

python0->>python0: load-checkpoint
python0->>+native0: register_state_dict(state_dict)
python0->>python0: iteration
python0->>+gpu0: forward(1)

native0->>+Daemon0: start
native0->>Daemon0: activate session
Note over Daemon0,Daemon1: network (tcp)
Daemon1->>Daemon0: recieve_state_registry(node:1[...])

native0->>+python0: read_scalar_state(..)
deactivate python0
native0->>+python0: read_blob_state(..)
deactivate python0
native0->>+python0: read_tensor_medata(..)
deactivate python0
python0->>native0: finalize(state_dict)

python0->>gpu0: backward(1)
python0->>gpu0: optimizer::step(1)

native0->>+Daemon0: local_state_registry
Note over native0,Daemon0: tcp
deactivate native0
DaemonN->>Daemon0: recieve_state_registry(node:1[...])
Daemon0->Daemon0: Global-Plan
Daemon0->>Daemon1: broadcast_metadata[tree]
Daemon0->>DaemonN: broadcast_metadata[tree]

Daemon0->>native0: snapshot-plan
gpu0->>-python0: iteration:0 complete
python0->>python0: save-checkpoint
python0->>python0: update-state-dict
python0->>+native0: register_state_dict(state_dict)

DaemonN->>Daemon0: snapshot-0-finalized
native0->>Daemon0: activate-snapshot
Daemon1->>Daemon0: snapshot-0-finalized
python0->>native0: snapshot-scalars+blobs(state_dict)
python0->>native0: save_async()
gpu0->>native0: snapshot-tensors
native0-->>python0: async-complete
native0->>Daemon0: local-snapshot
deactivate native0
Note over native0,Daemon0: shared_memory

Daemon0->>Daemon0: write_ckpt_data_file
Daemon0->>Daemon0: write_ckpt_index_file
Daemon0->>Daemon1: broadcast: snapshot-0-complete
Daemon0->>DaemonN: broadcast: snapshot-0-complete
deactivate Daemon0

activate python0


deactivate python0
Daemon0->>DaemonN: relay_metadata(node:1[...])



python0->>+gpu0: forward(1)
python0->>gpu0: backward(1)
python0->>gpu0: optimizer::step(1)

Daemon0->>Daemon1: broadcast_metadata[tree]
Daemon0->>DaemonN: broadcast_metadata[tree]

native0<<-->>+gpu0: read_tensor_data()
python0->>gpu0: synchronize(1)
deactivate Daemon0
native0->>+Daemon0: send_tensor_data[tree]
native0<<-->>gpu0: read_tensor_data()
deactivate gpu0
native0->>Daemon0: send_tensor_data[tree]
native0-->>gpu0: optimizer_raise_event()
native0->>Daemon0: finalize_checkpoint()
Daemon0->Daemon0: Serialize
Daemon0->>Daemon1: broadcast_notify_checkpoint[tree]
Daemon0->>DaemonN: broadcast_notify_checkpoint[tree]
Daemon0-->>-native0: on_checkpoint_complete()
native0-->>python0: on_checkpoint_complete()

gpu0->>gpu0: optimizer:start
gpu0-->>-python0: iteration:1 complete

activate Daemon0

```

```

```

```

```
