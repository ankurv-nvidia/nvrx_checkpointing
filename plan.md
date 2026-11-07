# In-process checkpoint daemon

## Agreed behavior

- Work on branch `inproc`, created from fetched `origin/main`, in its separate
  worktree.
- Define `NATIVE_CKPT_SERVER_INPROC` in `src/config.h`. Its presence selects
  in-process mode; absence preserves separate-process startup.
- In-process mode hosts the node daemon in the local-rank-0 training process.
  Other local ranks connect through the existing HTTP and shared-memory paths.
  A conflicting separate-process daemon produces a clear startup error.
- Use the same reference lifetime model in both daemon modes.
- A training client acquires a local session reference associated with its
  process identity. It retains that reference until its work and client worker
  threads finish, then explicitly releases it. A dead process invalidates its
  remaining references automatically.
- The training-side root owns the client reference. Generic state-tree copies,
  deserialization, and daemon-owned trees do not acquire client references.
- A node acquires one reference for its session on the coordinator when its
  first local reference appears. The coordinator reference remains held while
  the local session has valid references, including unfinished daemon work.
  It is explicitly released when those references are gone and automatically
  invalidated when a service ping to that node fails.
- Every asynchronous daemon task and pending web API call holds a daemon
  self-reference until it finishes. References carry their owner identity and
  are independent of task semantics.
- The reaper checks self-references first, then local process references, then
  remote service references. It stops at the first valid reference and keeps
  the daemon alive. Self-references are always valid.
- A newly started daemon allows approximately five seconds to acquire its
  first reference. After use, absence of valid references initiates shutdown
  immediately. Clean shutdown has no five-second grace delay.
- Task timeout and failure handling are separate from reference reaping.
  Auditing and changing existing task timeouts is a separate follow-up.
- The training main thread waits for its hosted daemon to shut down. The
  daemon lifetime thread runs reaping and shutdown, and joins the daemon's own
  threads. The daemon does not inspect training objects or manage training
  client worker threads.
- Separate-process startup retains its process-wide setup and stderr
  redirection. In-process startup does not redirect training streams or install
  daemon-specific process-wide signal handlers. It writes daemon logs directly
  to `NATIVE_CKPT_SERVER_LOG`.
- The CI launcher prestarts a daemon only for valgrind. Other flavors leave
  startup to the training client. A new `inproc` flavor enables
  `NATIVE_CKPT_SERVER_INPROC` in the training environment.

## Implementation

1. Add configuration and a reference registry with explicit acquire/release,
   process identity checks, remote service validation, and the ordered reaper.
   Make acquisition/release retries safe and associate references with sessions.
2. Add local client and node-to-coordinator reference endpoints. Keep upstream
   references alive through session self-references and forwarding handlers.
   Reuse existing session topology and checkpoint completion communication.
3. Add scoped self-reference ownership around daemon asynchronous operations
   and web requests, including delegated completion forwarding. Serialize the
   final reference decision against new request admission and acquisitions.
4. Add an explicitly owned production in-process daemon lifetime thread.
   Start only on local rank 0, wait for readiness on other local ranks, detect
   daemon ownership conflicts, and join full shutdown before the owner exits.
   Preserve the existing multi-instance test/benchmark helpers.
5. Attach a client session reference to the training-side root and release it
   after its client work finishes. Add automatic client-side exit cleanup before
   interpreter finalization; release the GIL while waiting. Keep the daemon
   independent of root objects and Python lifecycle details.
6. Link the daemon and checkpoint writer into production native consumers
   without circular CMake dependencies. Preserve CPU-only builds, editable
   installs, and strict development checks.
7. Route in-process daemon logs directly to the configured file, including raw
   performance markers. Keep logging and telemetry alive through daemon work
   and final collector flushing, with safe explicit shutdown ordering.
8. Document the opt-in mode and reference-driven lifetime in the repository.
9. Remove daemon prelaunch from ordinary CI flavors, retain valgrind's explicit
   server context, and add the `inproc` flavor with focused launcher and flavor
   coverage. Include the flavor in generated CI through the existing mechanism.

## Validation

- Reference registry checks: explicit release, dead process cleanup, self/PID/
  remote priority, stopping after the first valid reference, session isolation,
  and startup-only grace.
- Existing native checkpoint and coordinator durability tests.
- Independent training-process coverage: one daemon owner per node, non-owner
  readiness, no spawned daemon in in-process mode, and a clear ownership
  conflict with an existing separate daemon.
- Root destruction and actual interpreter exit with client and daemon work in
  flight; owner shutdown waits for remaining local clients and remote sessions.
- Delayed peer/coordinator completion and forwarding remain reachable until
  self-references are released; clean shutdown has no added five-second delay.
- Direct daemon file logging leaves training stderr usable. Exercise telemetry
  shutdown where the existing fixtures permit it.
- Run relevant CPU native tests and Python tests; report unavailable fixtures
  or platform-specific validation explicitly. Review the complete final diff
  and run `git diff --check` before declaring completion.
- Verify that ordinary flavors never prestart a daemon, valgrind retains its
  server lifecycle, and the `inproc` flavor propagates the mode to training.
- Validate the pushed branch through JET CI with the GB200 16-node
  `super3bf16` workload and the `perf+native+inproc` flavor.

## Scope control

Discuss changes to these agreed semantics before changing this plan. Routine
implementation choices and fixes that preserve these semantics are in scope.
No history rewrite, performance-baseline changes, or broad timeout audit is
part of this task.
