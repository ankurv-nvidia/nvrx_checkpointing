# In-process daemon

- Centralize self, PID, and remote references, validity checks, periodic reaping,
  coordinator references, and shutdown waiting in `service_reference.cpp`.
- Hold one reference for each request or asynchronous operation. Training roots
  hold PID references until their producer work and shared-memory queues drain.
- Start the daemon in local rank zero when `NATIVE_CKPT_SERVER_INPROC` is present.
  Join it during training-process exit; retain standalone launch behavior otherwise.
- Write in-process daemon logs directly to `NATIVE_CKPT_SERVER_LOG`.
- Keep the five-second grace period at startup only. Release idle session
  references promptly and retain coordinator access through final telemetry flush.
- Prestart the CI daemon only for valgrind; add the `inproc` flavor. Validate
  reference lifetimes locally and run the 16-node inproc workload through JET CI.
