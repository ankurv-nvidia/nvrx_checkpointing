# Plan/Milestones

## Native onboarding

### Code
* C++ model registry (sharded_state_dict)
* Custom (thin/low-code) reader/writer in C++
* OOP-ckpt-server for (local/remote) Disk-IO and networking (p2p)

### KPIs
* No change to model training infrastructure
* Bechmarking and SOL measurements

### Notes:
* Meant as a proof-concept milestone
* Enable benchmarking of CPU / GPU compute and memory utilization
* Iron out hurdles to C++ build-systems wrt. ease-of-shipping mixed code 
    * Determine minimum python / gcc / libstdc++ versions and requirements
* Iron out issues with optimizer stall apis (async-d2h-saves)

### Perf-Targets
* Checkpoint data registration (~10ms)
* Checkpoint data extraction (~ 300ms)
* Exposed overhead (~100ms)
    * Everything inclusive
    * including slowdown due to increased HBM activity during training

## Fast 1st Save

### Code
* DistCP format reader/writer in C++
    * File format/data parity with existing save (tensors)
    * Custom format for scalars/blobs/metadata storage/retrieval
* Deprecate (save) python code

### KPIs
* Saves ~5mins from recovery due to faster 1st and 2nd saves
* Enables more frequent saves
* Reduce Host memory overheads and improve perf jitter

### Notes:
* No change to load mechanisms
* Iron out issues with out-of-proc ckpt server logistics and code placement/shipping
    * Potentially bundling with System infra / nvrx
    * API surface between MLM and checkpointins (practicing backward compat)
* Deprecate Nvrx and MLM checkpointing code (python)
* Iron out issues with C++ build sytems

## Recovery @ SOL

### Code
* Persistent Checkpointing Server (in-memory) 
* Auto on-demand reshuffling

### KPIs
* Fast in-job recovery with hot-spares
* Deprecate the existing python code for async checkpointing from MLM and NVRx

## Load @ SOL

### Code
* TBD 

### KPIs
* Fast first load

### Notes: