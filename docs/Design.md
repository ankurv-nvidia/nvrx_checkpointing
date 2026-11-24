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

# Design Overview

The two figures follow one async save on rank 0, showing where each step runs and which network carries it: first in nvrx-python, as Megatron-LM runs it with `--async-save --use-persistent-ckpt-worker --ckpt-assume-constant-structure`, then in this design.

```mermaid
block-beta
  columns 1
  block:time
    columns 8
    tl["nvrx-python"] t_o["once per<br/>session"] t_o2[" "] t_s["save N"] t_s2[" "] t_t["training<br/>continues"] t_f["finalize"] t_f2["time →"]
  end
  block:ranks
    columns 8
    l_r["Other<br/>ranks"] space md["metadata<br/>×N"] space space space rs["write<br/>results ×N"] space
  end
  block:nccl
    columns 8
    l_n["NCCL"] n_o["all-gather,<br/>gather"] space space space n_t["all-reduce"] space n_f["gather,<br/>barrier"]
  end
  block:main
    columns 8
    l_m["Main thread<br/>rank 0"] sd0["state dict"] gp["global plan"] sd1["state dict"] sb["scalars,<br/>blobs"] space mf[("metadata file")] tr[("tracker")]
  end
  block:busyband
    columns 8
    l_b["Main thread<br/>busy"] b_o1[" "] b_o2[" "] b_s1[" "] b_s2["sync, wait<br/>for D2H"] b_t["per iteration"] b_f1[" "] b_f2[" "]
  end
  block:bg
    columns 8
    l_w["Background<br/>worker"] space space tb["tensor bytes"] space sf[("shard files")] wr["write<br/>results"] space
  end
  block:ipc
    columns 8
    l_s["Local<br/>IPC"] space space space space space space space
  end
  block:dm
    columns 8
    l_d["Daemon"] space space nd["no daemon"] space space space space
  end
  block:tcp
    columns 8
    l_h["Host<br/>TCP"] space space nt["not used"] space space space space
  end
  block:peers
    columns 8
    l_o["Other<br/>daemons"] space space space space space space space
  end
  sd0 --> gp
  md --> gp
  sd1 --> sb
  sd1 -- "CUDA IPC, blobs" --> tb
  tb --> sf
  sf --> wr
  wr --> mf
  rs --> mf
  class tl,l_r,l_n,l_m,l_b,l_w,l_s,l_d,l_h,l_o lbl
  class t_o,t_o2,t_t phA
  class t_s,t_s2,t_f,t_f2 phB
  class n_o,n_t,n_f,nd,nt note
  class sd0,sd1,sb py
  class md,gp,rs,wr meta
  class tb data
  class mf,tr,sf file
  class b_o1,b_o2,b_s1,b_s2,b_f1,b_f2 busy
  class b_t tick
  classDef lbl fill:none,stroke:none,font-weight:bold
  classDef note fill:none,stroke:none,color:#777777,font-style:italic
  classDef phA fill:#e8e8e8,stroke:none,color:#333333
  classDef phB fill:#f5f5f5,stroke:none,color:#333333
  classDef py fill:#e1d5e7,stroke:#9673a6,color:#000000
  classDef native fill:#d5e8d4,stroke:#82b366,color:#000000
  classDef meta fill:#dae8fc,stroke:#6c8ebf,color:#000000
  classDef data fill:#fff2cc,stroke:#d6b656,color:#000000
  classDef file fill:#f5f5f5,stroke:#666666,color:#000000
  classDef busy fill:#f39c3399,stroke:#b45f06,color:#3d1f00
  classDef tick fill:none,stroke:#b45f06,stroke-dasharray:3 3,color:#b45f06
  style time fill:none,stroke:none
  style ranks fill:#8080801a,stroke:none
  style nccl fill:none,stroke:#999999,stroke-dasharray:5 4
  style main fill:#ffa5001f,stroke:none
  style busyband fill:#ffa5001f,stroke:none
  style bg fill:#4a90e21f,stroke:none
  style ipc fill:none,stroke:#999999,stroke-dasharray:5 4
  style dm fill:#5cb85c1f,stroke:none
  style tcp fill:none,stroke:#999999,stroke-dasharray:5 4
  style peers fill:#8080801a,stroke:none
```

```mermaid
block-beta
  columns 1
  block:time
    columns 8
    tl["This design"] t_o["once per<br/>session"] t_o2[" "] t_s["save N"] t_s2[" "] t_t["training<br/>continues"] t_f["finalize"] t_f2["time →"]
  end
  block:ranks
    columns 8
    l_r["Other<br/>ranks"] space space space space space space space
  end
  block:nccl
    columns 8
    l_n["NCCL"] space space nn["unused by<br/>checkpoints"] space space space space
  end
  block:main
    columns 8
    l_m["Main thread<br/>rank 0"] sd0["state dict"] space sd1["state dict"] space space space tr[("tracker")]
  end
  block:busyband
    columns 8
    l_b["Main thread<br/>busy"] space space space space space b_f1[" "] space
  end
  block:bg
    columns 8
    l_l["Background<br/>library"] nc0["native clone"] space nc1["native clone"] space tb["tensor bytes"] space space
  end
  block:ipc
    columns 8
    l_s["Local<br/>IPC"] space space space space space space space
  end
  block:dm
    columns 8
    l_d["Coordinator<br/>daemon"] rg["registries"] gp["global plan"] space space sf[("shard files")] mf[("metadata file")] vd["verdict"]
  end
  block:tcp
    columns 8
    l_h["Host<br/>TCP"] space space mo["metadata<br/>only"] space space space space
  end
  block:peers
    columns 8
    l_o["Other<br/>daemons"] rgn["registries ×N"] pln["plans ×N"] space space space bsn["blob sizes<br/>×N"] vdn["verdict ×N"]
  end
  block:keyrow
    columns 8
    k0["Key"] k1["Python<br/>objects"] k2["native<br/>memory"] k3["metadata"] k4["tensor<br/>bytes"] k5[("file")] k6["main thread<br/>busy"] k7["×N: one per<br/>rank or node"]
  end
  sd0 -- "clone" --> nc0
  nc0 -- "registry" --> rg
  rgn --> rg
  rg --> gp
  gp --> pln
  sd1 -- "sync" --> nc1
  nc1 -- "D2H" --> tb
  tb -- "shared memory" --> sf
  sf --> mf
  bsn --> mf
  mf --> vd
  vd --> vdn
  vd -- "notify" --> tr
  class tl,l_r,l_n,l_m,l_b,l_l,l_s,l_d,l_h,l_o,k0 lbl
  class t_o,t_o2,t_t phA
  class t_s,t_s2,t_f,t_f2 phB
  class nn,mo,k7 note
  class sd0,sd1,k1 py
  class nc0,nc1,k2 native
  class rg,gp,rgn,pln,bsn,vd,vdn,k3 meta
  class tb,k4 data
  class sf,mf,tr,k5 file
  class b_f1,k6 busy
  classDef lbl fill:none,stroke:none,font-weight:bold
  classDef note fill:none,stroke:none,color:#777777,font-style:italic
  classDef phA fill:#e8e8e8,stroke:none,color:#333333
  classDef phB fill:#f5f5f5,stroke:none,color:#333333
  classDef py fill:#e1d5e7,stroke:#9673a6,color:#000000
  classDef native fill:#d5e8d4,stroke:#82b366,color:#000000
  classDef meta fill:#dae8fc,stroke:#6c8ebf,color:#000000
  classDef data fill:#fff2cc,stroke:#d6b656,color:#000000
  classDef file fill:#f5f5f5,stroke:#666666,color:#000000
  classDef busy fill:#f39c3399,stroke:#b45f06,color:#3d1f00
  classDef tick fill:none,stroke:#b45f06,stroke-dasharray:3 3,color:#b45f06
  style time fill:none,stroke:none
  style ranks fill:#8080801a,stroke:none
  style nccl fill:none,stroke:#999999,stroke-dasharray:5 4
  style main fill:#ffa5001f,stroke:none
  style busyband fill:#ffa5001f,stroke:none
  style bg fill:#4a90e21f,stroke:none
  style ipc fill:none,stroke:#999999,stroke-dasharray:5 4
  style dm fill:#5cb85c1f,stroke:none
  style tcp fill:none,stroke:#999999,stroke-dasharray:5 4
  style peers fill:#8080801a,stroke:none
  style keyrow fill:none,stroke:none
```

<details>
<summary>The same comparison as one draw.io figure</summary>

![nvrx-python and this design: where each step of an async save runs, and which network carries it](./Design-overview.drawio.svg)

</details>

**A native clone on library threads, the rest in the background.** The client library keeps a native clone of the state dict on its own threads. The first registration copies the scalars and blobs into native memory and records each tensor by device pointer, creating no Python objects, and each async save syncs the clone with the state dict in the background. Apart from passing the snapshot gates, which normally open at once, and a short callback at [gate 8.1](#stages), the main thread does no checkpoint work. Library threads also copy the tensors device-to-host (D2H) into the daemon's shared memory, the daemons plan once per session and make the shards durable, and the coordinator daemon writes the checkpoint metadata file.

PyTorch DCP also writes the metadata file in the background, but by default copies the state dict to host memory on the calling thread. nvrx-python overlaps only the shard writes with training: its main thread builds the global plan over NCCL and, on every save, syncs the scalars and blobs, waits for the D2H copy, and on rank 0 gathers every rank's results and writes the metadata file while the other ranks wait.

**The daemons' own network plane.** The daemons communicate asynchronously over TCP on the host. No checkpoint step uses the training process group, so ranks never enter a checkpoint collective, although the daemons share the host's CPU, memory bandwidth, PCIe and storage with training. On the save path only metadata crosses this network: registries and the global plan once per session, then each rank's status and blob sizes, from which the coordinator computes the metadata file's offsets and one verdict.

**Future work the design enables.** Fully asynchronous tensor capture removes the next optimizer step's wait for the D2H copy. Continuous in-memory checkpoints persist only some of the snapshots that finalize in daemon memory, so a rollback loses less work. Because out-of-proc daemons outlive the training processes, a restarted job can load from daemon memory instead of disk.

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

## TODO : Assumptions

- Daemon relies on megatron to validate the checkpoint file
- Shared - memory - Daemon
  - Add a Time sequence diagram for daemon and shared-memory

1. Daemon opens the shm file (listener)
   1.1 . shm file is already exists. it'll reinitialize it
   1.2 - Any producer thats still writing to the old file will error

# Daemon

The daemon operates as primarily a manager of:

1. Session : Captures all state that corresponds to a training-job
2. Lifetime Management : Process or In-Proc threads and the sessions
3. Messaging Infra: (Shared-Memory, Web-Service etc) : Relays requests/data to session(s) or processes lifetime management requests.

## Session

The session manages state that corresponds to a specific training-job. There are two main components of the state

1. Session-State : Computed once and applies to the entire session
2. Snapshot-State : Applies to each individual snapshot within a session as they are requested.

A session-state must be available and finalized before it begins to accept snapshots

### Session-State

The following are the most important session state

1. The local registry (per-local-rank)
2. Global Registry. A full detailed plan that knows the locations of each state-dictionary item as well as their replication or shard locations.
3. Deduped Local Registries (per-local-rank): This is also used as the snapshot plan and consists of rank-assignment to avoid duplicate snapshoting and disk-persistence for checkpoint state
4. Snapshots : Each snapshot further has its own state but the session maintains all the snapshots available within a session.
   1. Snapshots can be evicted to conserve memory
   2. Snapshots can be loaded (or preloaded) from disk as needed

A Session is considered initialized and valid once it has recieved the global registry from the coordinator (or merged all local registries on the coordinator) and computed the local-registries for its local-ranks
Any error during the session initialization will prevent a async-save from the training job and will be bubbled up during the first save request.

### Snapshot-State

A snapshot can be under-construction or finalized. Once finalize the snapshot state is considered frozen and not modified.
Snapshots can be evicted by the Session if the server starts running out of memory or based on user preference on how many active snapshots to maintain
Partial snapshots arent considered reusable. The following state is maintained in the snapshot

Once fine

1. Snapshot Registry
2. Byte-Storage

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

# Resiliency and Error Handling

Two principles shape how failures are handled:

1. **The daemon never intentionally crashes or raises.** It records each failure on the session or snapshot that the failure belongs to and reports it to the clients. A daemon crash is always a bug and is handled as a fatal fault (see [FT-launcher integration](#out-of-proc-mode--ft-launcher-integration-fault-tolerance)).
2. **The client API is almost entirely asynchronous.** Background work never raises into the training loop directly. Its errors travel through two _outlets_ and are raised only at synchronous points of the training loop.

## Stages

A _cycle_ is one launch of the training ranks, and every FT-launcher restart starts the next cycle. Each cycle has its own session in the daemon set (one daemon per node), and the session id is the cycle id. One daemon, the _coordinator_ (peer 0), builds the global plan and aggregates snapshot results for the whole job.

Stages 1-3 run once per cycle, without blocking the training loop:

1. **Daemon boot or attach.** The client library attaches to the node's daemon. The FT launcher starts the daemon, or the client starts it in self-managed mode.
2. **State-dict registration.** Each rank sends its local registry to its daemon, which forwards it to the coordinator.
3. **Global save-planning.** The coordinator merges every registry into the global plan and broadcasts it, and each daemon derives the deduplicated snapshot plans of its ranks.

Stages 4-8 run for every save:

4. **Snapshot activation.** The training loop blocks at three gates:
   - Gate 4.1 snapshot-plan-gate: waits for the rank's snapshot plan and raises any session error.
   - Gate 4.2 prev-snapshot-gate: waits until the a previous active snapshot is finalized and its callback has run, and raises that snapshot's error, if any.
     - <mark>TODO_CROSS_SESSION_GATE</mark> This also includes the "cross-session-gate" from the daemon that makes sure lingering active snapshots from zombied sessions on the same output location are \_quiescent\*: (finalized, with no write in flight).
       - <mark>OPEN-QUESTION</mark> Should we limit this to sessions with **Same output location.** Is it the checkpoint root or the per-iteration directory?

5. **Transfer.** Scalars and blobs are captured synchronously, before `save_checkpoint` returns. Tensors are copied device-to-host (D2H) in the background into the daemon's shared memory, and the next optimizer step waits on an event until the copy is done.
6. **Durability.** In the background, each daemon writes its ranks' shards and reports each rank's result to the coordinator.
7. **Disk finalization.** In the background, the coordinator waits for every rank's result, writes the metadata, and finalizes the snapshot with one _verdict_: succeeded, or failed with the collected errors.
8. **Notification.** The coordinator broadcasts the verdict and each daemon notifies its clients.
   - Gate 8.1 snapshot-finalization-gate: takes the GIL to run the client's finalization routine on the main python thread. Errors during Steps 4-8 are also propagated here.

## Error outlets

```mermaid
flowchart TB
    subgraph CYC ["Once per training cycle"]
        direction LR
        S1["1 Daemon boot<br/>or attach"] ==> S2["2 State-dict<br/>registration"] ==> S3["3 Global<br/>save-planning"]
    end
    subgraph SAVE ["Every save N"]
        direction LR
        S4["4 Snapshot activation<br/>gates 4.1, 4.2<br/>and cross-session"] ==> S5["5 Transfer<br/>D2H to the daemon"] ==> S6["6 Durability<br/>shards on disk"] ==> S7["7 Disk finalization<br/>one verdict"] ==> S8["8 Notification<br/>callback 8.1"]
    end
    O1{{"Outlet 1<br/>gate 4.1 raises<br/>inside save_checkpoint"}}
    O2{{"Outlet 2<br/>callback 8.1 hands<br/>a failed verdict over"}}
    TL["Training loop raises<br/>in save_checkpoint, at the<br/>next poll, or at gate 4.2"]
    PUB(["N published<br/>rank 0 writes the tracker"])
    FX["Daemon crash or hang<br/>or rank crash"]
    FT["FT launcher<br/>diagnostics and restart"]

    CYC ==>|"snapshot plans"| SAVE
    SAVE ==>|"success verdict"| PUB
    CYC -.->|"session errors<br/>stages 1-3"| O1
    SAVE -.->|"snapshot errors<br/>stages 4-8"| O2
    O1 -.->|"raised in the<br/>next save"| TL
    O2 -.->|"raised at the<br/>next sync point"| TL
    TL -.->|"Megatron policy<br/>rank exits"| FT
    FX -.->|"fatal, bypasses<br/>both outlets"| FT

    classDef outlet fill:paleturquoise,stroke:teal,color:black,stroke-width:3px
    classDef raise fill:mistyrose,stroke:firebrick,color:black
    classDef fatal fill:lightpink,stroke:darkred,color:black,stroke-width:2px
    classDef pub fill:honeydew,stroke:darkgreen,color:black
    class O1,O2 outlet
    class TL raise
    class FX,FT fatal
    class PUB pub
```

- **Outlet 1, gate 4.1** (synchronous, inside `save_checkpoint`) delivers session errors from stages 1-3. The coordinator records a planning failure on the session and broadcasts it, so every rank raises the same error at its next save.
- **Outlet 2, callback 8.1** (asynchronous) delivers snapshot errors from stages 4-8. Daemons record failures on the snapshot and report them to the coordinator, which aggregates them into the job-wide verdict. Callback 8.1 hands a failed verdict to the training loop, which raises it at its next sync point: the per-iteration poll (`maybe_finalize_async_save`) or gate 4.2 of the next save.
- **Fatal.** A daemon crash or hang, or a rank crash, bypasses both outlets. The FT launcher collects diagnostics and restarts training as the next cycle.

A snapshot is _active_ until stage 7 finalizes it. Only callback 8.1 of a succeeded snapshot _publishes_ it as the checkpoint to resume from (for Megatron, rank 0 writes the latest-checkpoint tracker file). A failed or partial snapshot is never published or reused, so the last published checkpoint stays the recovery point.

## How an error reaches every rank

```mermaid
sequenceDiagram
    participant R as Rank R
    participant LD as Rank R's<br/>daemon
    participant CO as Coordinator<br/>(peer 0)
    participant PD as Peer<br/>daemons

    R->>LD: [4] activate N<br/>[5] scalars, blobs
    Note right of R: save returns,<br/>training continues
    R->>LD: [5] tensors D2H<br/>in the background
    LD->>CO: [6] rank R's result
    PD->>CO: [6] other results
    alt success
        CO->>CO: [7] every result OK,<br/>metadata written,<br/>N succeeded
    else failure
        CO->>CO: [7] a daemon recorded<br/>an error, N failed
    end
    CO->>PD: [8] one verdict,<br/>tree relay
    CO->>LD: [8] same verdict
    LD->>R: [8] notify clients
    alt N succeeded
        rect rgba(22,163,74,0.12)
            R->>R: [8.1] publish N
        end
    else N failed
        rect rgba(220,38,38,0.12)
            R->>R: [8.1] hand the error<br/>to the training loop
            Note right of R: next poll or gate 4.2<br/>raises it (Outlet 2)
        end
    end
    Note over R,PD: Every rank receives the same verdict.<br/>A D2H failure on rank R never reaches<br/>a daemon, so only rank R learns of it,<br/>and N cannot finalize without rank R's<br/>result (see Open questions).
```

An error that occurs in or reaches a daemon becomes job-wide: the coordinator aggregates it into the snapshot's verdict, and every rank receives the same error. A planning failure reaches every rank's gate 4.1 the same way. An error that never leaves the client library, such as a D2H failure on rank R, reaches rank R only. The snapshot cannot finalize without rank R's result, and nothing bounds that wait (see [Open questions](#open-questions)).

## Concurrency

### Overlapping snapshots in one session

```mermaid
sequenceDiagram
    participant TL as Training<br/>loop
    participant CL as Client<br/>library
    participant D as Daemons

    Note over TL,D: N-1 is published, the recovery point
    TL->>CL: save N
    CL->>D: [4] activate N<br/>[5] scalars, blobs
    CL-->>TL: save N returns
    par training continues
        TL->>CL: per-iteration poll
        CL-->>TL: nothing to raise
    and snapshot N stays active
        CL->>D: [5] tensors D2H
        D->>D: [6] write shards
    end
    break N fails before save N+1
        D->>CL: [7, 8] N failed
        CL->>CL: [8.1] callback takes<br/>N's error
        CL-->>TL: next poll raises it,<br/>the rank exits
    end
    TL->>CL: save N+1
    rect rgba(255,165,0,0.12)
        Note over TL,CL: gate 4.2 blocks until N<br/>is finalized and its<br/>callback has run
        D->>CL: [7, 8] N's verdict
    end
    alt N succeeded
        CL->>CL: [8.1] publish N
        CL->>D: [4] activate N+1
        CL-->>TL: save N+1 returns
    else N failed
        CL->>CL: [8.1] callback takes<br/>N's error
        CL-->>TL: save N+1 raises it
        Note over TL,D: N+1 never activates. Megatron exits,<br/>and the FT launcher restarts from N-1
    end
```

N-1, N and N+1 denote consecutive saves. Once `save_checkpoint` returns, the tensor copy of stage 5 and stages 6-8 of save N overlap training. When save N+1 starts before snapshot N is finalized, gate 4.2 blocks the training loop until N is finalized and its callback has run. At most one snapshot per session is active, so the failures of N and N+1 never mix.

- **N's verdict arrives first.** On success, the callback publishes N and gate 4.2 passes at once. On failure, the next per-iteration poll raises N's error.
- **Save N+1 arrives first.** Gate 4.2 waits for N. On success, N is published and N+1 activates. On failure, save N+1 raises N's error and N+1 never activates.
- **Policy.** Gate 4.2 requires N to be finalized, not successful. Whether training continues after a failure is the client's policy. Megatron raises, so the rank exits and the FT launcher restarts training from N-1, the last published checkpoint.
- **Synchronous saves** wait for their own callback inside `save_checkpoint`, so they never overlap.

### Overlapping sessions: a new cycle while the previous one winds down

```mermaid
sequenceDiagram
    participant FT as FT<br/>launcher
    participant R as Ranks
    box Daemons, reused across cycles
        participant SK as Session k
        participant SK1 as Session k+1
    end

    R->>SK: in cycle k, save N<br/>activates S_k(N)
    R-xFT: rank crash
    FT->>FT: collect cycle k<br/>diagnostics
    FT->>R: stop cycle k,<br/>start cycle k+1
    Note over R: no callback 8.1 ran,<br/>N is not published
    par cycle k+1 sets up
        R->>R: load N-1
        R->>SK1: [1, 2] attach,<br/>register
        SK1-->>R: [3] snapshot plan
        R->>R: train to N
    and session k winds down
        SK->>SK: S_k(N) stages<br/>6 and 7 continue
    end
    R->>SK1: save N, gates<br/>4.1 and 4.2 pass
    rect rgba(255,165,0,0.12)
        SK1->>SK: cross-session gate,<br/>same output location
        Note over R: save N blocks
        SK-->>SK1: S_k(N) quiescent
    end
    SK1-->>R: S_k+1(N) activates
    Note over SK,SK1: no client is left in<br/>session k, so its verdict<br/>is only logged
    Note over R,SK1: if a rank died before its data reached<br/>a daemon, S_k(N) never finalizes and<br/>save N blocks forever (see Open questions)
```

Cycle k crashes while S_k(N), its snapshot of save N, is still active: durability is still running, or the coordinator is still waiting for results from ranks that died. The FT launcher collects cycle k's diagnostics and restarts the ranks as cycle k+1, reusing the daemons that did not crash.

- **Insulation.** Cycle k+1 runs stages 1-3 in its own session k+1, with a fresh plan, snapshots and error state, while session k winds down in the same daemon processes.
- **Recovery point.** No callback runs for S_k(N), so it is never published, and cycle k+1 resumes from N-1.
- **Same output location.** When cycle k+1 repeats save N, gates 4.1 and 4.2 pass, but the new snapshot writes to the same location as S_k(N). The cross-session gate holds its activation until S_k(N) is quiescent.
- **Errors of the old session.** Session k has no client left, so its verdict and errors are only logged and never raised into cycle k+1. The FT launcher redirects a reused daemon's log to the current cycle, so a late verdict of session k lands in cycle k+1's log.
- **Reuse.** Completed snapshots of session k may stay in daemon memory for a future fast load. Partial snapshots are never reused.

### Open questions

- <mark>TODO</mark> **Client-side failures.** Proposal: the client library reports a client-side failure (a registration or D2H error, for example) to its daemon whenever the daemon is reachable, so the coordinator fails the plan or the snapshot for every rank at once instead of waiting for input that never arrives.

### Errors by stage

Each timeline follows one error through a job on two nodes. Rank 0 and daemon 0, the coordinator, run on node 0. Rank 1 and daemon 1 run on node 1. The job is in cycle k, so its session in the daemons is session k, and N-1 is its last published checkpoint. Red bars and diamonds mark the error's path. The numbers give the order of its steps, and steps that happen at the same time share a number. Light-blue bars mark waits and background work, and grey diamonds mark a rank that the FT launcher stops. Time runs left to right and is not to scale.

A rank that raises usually exits, and the FT launcher then stops the other ranks and restarts training from N-1. The timelines show where each rank raises the error if the FT launcher has not stopped it first.

#### Stage 1 : Rank 1 cannot attach to its daemon

```mermaid
%%{init: {"gantt": {"fontSize": 16, "sectionFontSize": 16, "barHeight": 26, "barGap": 6, "leftPadding": 110, "topPadding": 50}}}%%
gantt
    dateFormat HH:mm
    axisFormat t+%M
    todayMarker off

    section Rank 0
    Load N-1                                  :r0a, 00:00, 2m
    Attach, register (async)                  :r0b, 00:02, 1m
    Train                                     :r0c, 00:03, 4m
    Save N waits at 4.1                       :active, r0d, 00:07, 3m
    (4) 4.1 snapshot-plan-gate raises         :crit, milestone, r0e, 00:10, 0m

    section Daemon 0<br>coordinator
    Waits for every registry                  :active, d0a, 00:03, 6m
    (3) Times out, fails the plan, broadcasts :crit, d0b, 00:09, 1m

    section Daemon 1
    (4) Gets the plan failure                 :crit, d1a, 00:10, 1m

    section Rank 1
    Load N-1                                  :r1a, 00:00, 2m
    (1) Attach fails                          :crit, milestone, r1b, 00:02, 0m
    (2) Register raises it                    :crit, r1c, 00:02, 1m
```

For example, rank 1 cannot map daemon 1's shared memory. Rank 1's client library already holds that error, so rank 1 raises it when it tries to send its local registry, and the registry never reaches the coordinator. The coordinator's registry wait times out, and it fails the plan for the whole job with a timeout error that names rank 1. Rank 0 blocks at gate 4.1 snapshot-plan-gate in save N until that failure arrives, then raises it.

#### Stage 2 : Daemon 1 rejects rank 1's registry

```mermaid
%%{init: {"gantt": {"fontSize": 16, "sectionFontSize": 16, "barHeight": 26, "barGap": 6, "leftPadding": 110, "topPadding": 50}}}%%
gantt
    dateFormat HH:mm
    axisFormat t+%M
    todayMarker off

    section Rank 0
    Load N-1                                  :r0a, 00:00, 2m
    Attach, register (async)                  :r0b, 00:02, 1m
    Train                                     :r0c, 00:03, 9m
    (4) Save N, 4.1 snapshot-plan-gate raises :crit, milestone, r0d, 00:12, 0m

    section Daemon 0<br>coordinator
    Waits for every registry                  :active, d0a, 00:03, 1m
    (2) Fails the plan, broadcasts            :crit, d0b, 00:04, 1m

    section Daemon 1
    (1) Rejects rank 1's registry             :crit, d1a, 00:03, 1m
    (3) Relays the plan failure               :crit, d1b, 00:05, 1m

    section Rank 1
    Load N-1                                  :r1a, 00:00, 2m
    Attach, register (async)                  :r1b, 00:02, 1m
    Train                                     :r1c, 00:03, 9m
    (4) Save N, 4.1 snapshot-plan-gate raises :crit, milestone, r1d, 00:12, 0m
```

For example, the registry holds a tensor type that the daemon does not support. Daemon 1 rejects the registry after the register call has returned, so neither rank sees the error before save N. Daemon 1 records the rejection on session k and reports it, the coordinator fails the plan, and both ranks raise the same error at gate 4.1 snapshot-plan-gate.

#### Stage 3 : The coordinator cannot merge the registries

```mermaid
%%{init: {"gantt": {"fontSize": 16, "sectionFontSize": 16, "barHeight": 26, "barGap": 6, "leftPadding": 110, "topPadding": 50}}}%%
gantt
    dateFormat HH:mm
    axisFormat t+%M
    todayMarker off

    section Rank 0
    Load N-1                                  :r0a, 00:00, 2m
    Attach, register (async)                  :r0b, 00:02, 1m
    Train                                     :r0c, 00:03, 9m
    (4) Save N, 4.1 snapshot-plan-gate raises :crit, milestone, r0d, 00:12, 0m

    section Daemon 0<br>coordinator
    Merges the registries                     :active, d0a, 00:03, 2m
    (1) Merge fails                           :crit, milestone, d0b, 00:05, 0m
    (2) Fails the plan, broadcasts            :crit, d0c, 00:05, 1m

    section Daemon 1
    Forwards rank 1's registry                :active, d1a, 00:03, 1m
    (3) Relays the plan failure               :crit, d1b, 00:06, 1m

    section Rank 1
    Load N-1                                  :r1a, 00:00, 2m
    Attach, register (async)                  :r1b, 00:02, 1m
    Train                                     :r1c, 00:03, 9m
    (4) Save N, 4.1 snapshot-plan-gate raises :crit, milestone, r1d, 00:12, 0m
```

For example, rank 0 and rank 1 register the same tensor with different global shapes. The coordinator records the planning failure once and broadcasts it in place of the plan, and both ranks raise it at gate 4.1 snapshot-plan-gate in save N.

#### Stage 4 : Daemon 1 rejects the activation of snapshot N

```mermaid
%%{init: {"gantt": {"fontSize": 16, "sectionFontSize": 16, "barHeight": 26, "barGap": 6, "leftPadding": 110, "topPadding": 50}}}%%
gantt
    dateFormat HH:mm
    axisFormat t+%M
    todayMarker off

    section Rank 0
    Save N, 4.1 and 4.2 pass                  :r0a, 00:00, 1m
    Train, D2H in background                  :r0b, 00:01, 9m
    (4) 8.1 snapshot-finalization-gate        :crit, r0c, 00:09, 1m
    (5) Next poll raises N's error            :crit, milestone, r0d, 00:10, 0m

    section Daemon 0<br>coordinator
    Writes rank 0's shards                    :active, d0a, 00:01, 7m
    (3) Fails N, broadcasts                   :crit, d0b, 00:08, 1m

    section Daemon 1
    (1) Rejects activation of N               :crit, milestone, d1a, 00:01, 0m
    (2) Records it, tells daemon 0            :crit, d1b, 00:01, 1m
    (4) Relays the verdict                    :crit, d1c, 00:09, 1m

    section Rank 1
    Save N, 4.1 and 4.2 pass                  :r1a, 00:00, 1m
    Train                                     :r1b, 00:01, 9m
    (5) 8.1 snapshot-finalization-gate        :crit, r1c, 00:10, 1m
    (6) Next poll raises N's error            :crit, milestone, r1d, 00:11, 0m
```

For example, daemon 1 cannot reserve shared memory for snapshot N. Save N still returns on both ranks. Daemon 1 records the rejection on snapshot N and tells the coordinator that rank 1 failed, and the coordinator finalizes N as failed once it has rank 0's result too. Gate 8.1 snapshot-finalization-gate hands N's error to each training loop, which raises it at its next per-iteration poll.

#### Stage 5 : Rank 1's D2H copy fails

```mermaid
%%{init: {"gantt": {"fontSize": 16, "sectionFontSize": 16, "barHeight": 26, "barGap": 6, "leftPadding": 110, "topPadding": 50}}}%%
gantt
    dateFormat HH:mm
    axisFormat t+%M
    todayMarker off

    section Rank 0
    Save N, 4.1 and 4.2 pass                  :r0a, 00:00, 1m
    Train, D2H in background                  :r0b, 00:01, 10m
    (3) 8.1 snapshot-finalization-gate        :crit, r0c, 00:11, 1m
    (4) Next poll raises N's error            :crit, milestone, r0d, 00:12, 0m

    section Daemon 0<br>coordinator
    Writes rank 0's shards                    :active, d0a, 00:01, 3m
    Waits for rank 1                          :active, d0b, 00:04, 6m
    (2) Times out, fails N, broadcasts        :crit, d0c, 00:10, 1m

    section Daemon 1
    Gets no data from rank 1                  :active, d1a, 00:01, 9m
    (3) Relays the verdict                    :crit, d1b, 00:11, 1m

    section Rank 1
    Save N, 4.1 and 4.2 pass                  :r1a, 00:00, 1m
    Train, D2H in background                  :r1b, 00:01, 11m
    (1) D2H fails, client keeps it            :crit, milestone, r1c, 00:02, 0m
    (4) 8.1 snapshot-finalization-gate        :crit, r1d, 00:12, 1m
    (5) Next poll raises N's error            :crit, milestone, r1e, 00:13, 0m
```

Rank 1's client library keeps the D2H error, so no daemon learns of it and rank 1's result never reaches the coordinator. The coordinator's result wait times out, and it finalizes N as failed with a timeout error that names rank 1. Gate 8.1 snapshot-finalization-gate hands that failure to each training loop, which raises it at its next poll.

#### Stage 6 : Daemon 1 cannot write rank 1's shards

```mermaid
%%{init: {"gantt": {"fontSize": 16, "sectionFontSize": 16, "barHeight": 26, "barGap": 6, "leftPadding": 110, "topPadding": 50}}}%%
gantt
    dateFormat HH:mm
    axisFormat t+%M
    todayMarker off

    section Rank 0
    Save N, 4.1 and 4.2 pass                  :r0a, 00:00, 1m
    Train                                     :r0b, 00:01, 5m
    Save N+1 waits at 4.2                     :active, r0c, 00:06, 4m
    (4) 8.1 snapshot-finalization-gate        :crit, r0d, 00:09, 1m
    (5) 4.2 prev-snapshot-gate raises         :crit, milestone, r0e, 00:10, 0m

    section Daemon 0<br>coordinator
    Writes rank 0's shards                    :active, d0a, 00:01, 7m
    (3) Fails N, broadcasts                   :crit, d0b, 00:08, 1m

    section Daemon 1
    Writes rank 1's shards                    :active, d1a, 00:01, 3m
    (1) Shard write fails                     :crit, milestone, d1b, 00:04, 0m
    (2) Records it, tells daemon 0            :crit, d1c, 00:04, 1m
    (4) Relays the verdict                    :crit, d1d, 00:09, 1m

    section Rank 1
    Save N, 4.1 and 4.2 pass                  :r1a, 00:00, 1m
    Train                                     :r1b, 00:01, 5m
    Save N+1 waits at 4.2                     :active, r1c, 00:06, 5m
    (5) 8.1 snapshot-finalization-gate        :crit, r1d, 00:10, 1m
    (6) 4.2 prev-snapshot-gate raises         :crit, milestone, r1e, 00:11, 0m
```

For example, node 1's file system is full. Both ranks call save N+1 before N's verdict arrives, so gate 4.2 prev-snapshot-gate blocks them. When the verdict comes, gate 8.1 snapshot-finalization-gate hands over N's error, gate 4.2 raises it on each rank, and N+1 never activates. Gate 8.1 can run during the gate 4.2 wait only once the GIL question in [Open questions](#open-questions) is settled.

#### Stage 7 : The coordinator cannot write the metadata

```mermaid
%%{init: {"gantt": {"fontSize": 16, "sectionFontSize": 16, "barHeight": 26, "barGap": 6, "leftPadding": 110, "topPadding": 50}}}%%
gantt
    dateFormat HH:mm
    axisFormat t+%M
    todayMarker off

    section Rank 0
    Save N, 4.1 and 4.2 pass                  :r0a, 00:00, 1m
    Train                                     :r0b, 00:01, 9m
    (3) 8.1 snapshot-finalization-gate        :crit, r0c, 00:09, 1m
    (4) Next poll raises N's error            :crit, milestone, r0d, 00:10, 0m

    section Daemon 0<br>coordinator
    Writes rank 0's shards                    :active, d0a, 00:01, 5m
    Writes the metadata                       :active, d0b, 00:06, 2m
    (1) Write fails                           :crit, milestone, d0c, 00:08, 0m
    (2) Fails N, broadcasts                   :crit, d0d, 00:08, 1m

    section Daemon 1
    Writes rank 1's shards                    :active, d1a, 00:01, 5m
    (3) Relays the verdict                    :crit, d1b, 00:09, 1m

    section Rank 1
    Save N, 4.1 and 4.2 pass                  :r1a, 00:00, 1m
    Train                                     :r1b, 00:01, 9m
    (4) 8.1 snapshot-finalization-gate        :crit, r1c, 00:10, 1m
    (5) Next poll raises N's error            :crit, milestone, r1d, 00:11, 0m
```

The coordinator's metadata write fails after every shard is on disk, and N is unusable without its metadata. The coordinator finalizes N as failed with its own error, and both ranks raise it at their next poll.

#### Stage 8 : Rank 0's finalization routine fails

```mermaid
%%{init: {"gantt": {"fontSize": 16, "sectionFontSize": 16, "barHeight": 26, "barGap": 6, "leftPadding": 110, "topPadding": 50}}}%%
gantt
    dateFormat HH:mm
    axisFormat t+%M
    todayMarker off

    section Rank 0
    Save N, 4.1 and 4.2 pass                  :r0a, 00:00, 1m
    Train                                     :r0b, 00:01, 8m
    (1) 8.1 snapshot-finalization-gate fails  :crit, r0c, 00:09, 1m
    (2) Next poll raises the error            :crit, milestone, r0d, 00:10, 0m

    section Daemon 0<br>coordinator
    Writes rank 0's shards                    :active, d0a, 00:01, 5m
    Writes the metadata                       :active, d0b, 00:06, 2m
    N succeeded, broadcasts                   :d0c, 00:08, 1m

    section Daemon 1
    Writes rank 1's shards                    :active, d1a, 00:01, 5m
    Relays the verdict                        :d1b, 00:09, 1m

    section Rank 1
    Save N, 4.1 and 4.2 pass                  :r1a, 00:00, 1m
    Train                                     :r1b, 00:01, 11m
    8.1 snapshot-finalization-gate            :r1c, 00:10, 1m
    (3) FT launcher stops it                  :done, milestone, r1d, 00:12, 0m
```

N succeeds, but gate 8.1 snapshot-finalization-gate fails on rank 0, for example because rank 0 cannot write the latest-checkpoint tracker file. N is on disk but never published. Only rank 0 raises the error. When rank 0 exits, the FT launcher stops rank 1 and restarts training from N-1.

#### Fatal : Daemon 1 crashes

```mermaid
%%{init: {"gantt": {"fontSize": 16, "sectionFontSize": 16, "barHeight": 26, "barGap": 6, "leftPadding": 110, "topPadding": 50}}}%%
gantt
    dateFormat HH:mm
    axisFormat t+%M
    todayMarker off

    section Rank 0
    Save N, 4.1 and 4.2 pass                  :r0a, 00:00, 1m
    Train                                     :r0b, 00:01, 6m
    (3) FT launcher stops it                  :done, milestone, r0c, 00:07, 0m

    section Daemon 0<br>coordinator
    Writes rank 0's shards                    :active, d0a, 00:01, 5m
    Waits for rank 1                          :active, d0b, 00:06, 5m
    (5) Times out, fails N, logs it           :crit, d0c, 00:11, 1m

    section Daemon 1
    Writes rank 1's shards                    :active, d1a, 00:01, 3m
    (1) Crashes                               :crit, milestone, d1b, 00:04, 0m

    section Rank 1
    Save N, 4.1 and 4.2 pass                  :r1a, 00:00, 1m
    Train                                     :r1b, 00:01, 6m
    (3) FT launcher stops it                  :done, milestone, r1c, 00:07, 0m

    section FT launcher
    (2) Collects diagnostics                  :crit, ft1, 00:04, 3m
    (4) Restarts from N-1                     :ft2, 00:07, 5m
```

No outlet carries a crash. The FT launcher collects diagnostics, stops both ranks and restarts training as cycle k+1 from N-1. In session k, the coordinator's wait for rank 1's result later times out and finalizes N as failed, which is only logged because session k has no client left.

See [Overlapping sessions](#overlapping-sessions-a-new-cycle-while-the-previous-one-winds-down).

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
