# In-process daemon

- Centralize self, PID, and remote references, validity checks, periodic reaping,
  coordinator references, and shutdown waiting in `service_reference.cpp`.
- Hold one reference for each request or asynchronous operation. Training roots
  hold PID references until their producer work and shared-memory queues drain.
- Run in-process by default; any rank can win ownership. Set
  `NATIVE_CKPT_SERVER_DISABLE_INPROC=1` for a separate process.
- Acquire process-lifetime telemetry references and join the owned daemon in
  native C++. Forked children cannot release or join their parent's resources.
- Retain failed remote references for later checks; a failed probe or changed
  startup generation makes a reference ineligible for the current keep-alive check.
- Write in-process daemon logs directly to `NATIVE_CKPT_SERVER_LOG`.
- Keep the five-second grace period at startup only. Release idle session
  references promptly and retain coordinator access through final telemetry flush.
- Prestart the CI daemon only for valgrind; add the `oop` flavor. Validate
  reference lifetimes locally and both modes through 16-node JET CI workloads.
