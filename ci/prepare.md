# CI Runner Setup on Cluster Login Nodes

All state — token, config, binary, logs — lives under `/home/ankurv/data/ci`.
Lock down access once: `chmod 700 /home/ankurv/data/ci`

Script: `megatron/native/ci/runner/setup_runner.py`

---

## Directory layout

```
/home/ankurv/data/ci/
├── token           ← registration token (you write this, chmod 600)
├── config          ← optional settings: GITLAB_URL, CLUSTER, CONCURRENT
├── gitlab-runner   ← binary (downloaded by setup)
├── config.toml     ← runner config (written by setup after registration)
├── runner.pid      ← PID of running process
├── runner.log      ← runner log output
├── builds/         ← job build directories
└── cache/          ← job cache
```

---

## 1. Get a registration token

1. Go to **https://gitlab-master.nvidia.com/ankurv/\<project\>/-/settings/ci_cd**
2. Expand **Runners → New project runner**
3. Set a description (e.g. `cwdfw login node`) and tag (e.g. `slurm-cwdfw`), click **Create runner**
4. Copy the token — it starts with `glrt-`

> The token is **single-use**. After `setup` succeeds, it is exchanged for a persistent auth token stored in `config.toml`. If a previous `setup` attempt failed (e.g. due to wrong URL), the token may already be consumed — go back to GitLab and create a fresh one.

---

## 2. Write config files (once per cluster)

SSH into the login node, then:

```bash
CI_DIR=/home/ankurv/data/ci
mkdir -p $CI_DIR
chmod 700 $CI_DIR

# Write the registration token
echo 'glrt-xxxxxxxxxxxxxxxxxxxx' > $CI_DIR/token
chmod 600 $CI_DIR/token

# Write settings (CLUSTER is auto-detected from hostname if omitted)
cat > $CI_DIR/config <<EOF
GITLAB_URL=https://gitlab-master.nvidia.com
CLUSTER=cwdfw        # required — set explicitly, e.g. cwdfw, ocihsg, eos
CONCURRENT=4
EOF
chmod 600 $CI_DIR/config
```

For a second cluster (run this on that cluster's login node with its own token):
```bash
echo 'glrt-yyyyyyyyyyyyyyyyyyyy' > /home/ankurv/data/ci/token
chmod 600 /home/ankurv/data/ci/token
# Update CLUSTER= in /home/ankurv/data/ci/config accordingly
```

---

## 3. Register and start (first time only)

```bash
python3 megatron/native/ci/runner/setup_runner.py setup
```

This does three things:
1. Downloads `gitlab-runner` binary into `/home/ankurv/data/ci/`
2. Registers with GitLab and writes `config.toml`
3. Starts the runner process (PID saved to `runner.pid`, output to `runner.log`)

---

## 4. Start / stop / status

```bash
python3 setup_runner.py start      # start after a reboot or stop
python3 setup_runner.py stop       # graceful stop (SIGTERM, then SIGKILL after 15s)
python3 setup_runner.py status     # shows PID + last 20 lines of runner.log
tail -f /home/ankurv/data/ci/runner.log   # live log stream
```

---

## 5. Clean up everything

```bash
python3 setup_runner.py cleanup --yes
```

This stops the process, calls `gitlab-runner unregister`, deletes `config.toml`, `runner.pid`, `runner.log`, `gitlab-runner` binary, `builds/`, and `cache/`.

The `token` file is kept so you can re-register without fetching a new token.
Remove it manually when done:

```bash
rm /home/ankurv/data/ci/token /home/ankurv/data/ci/config
```

---

## 6. Targeting a cluster in CI

All values are hardcoded in `gitlab-ci-slurm.yml`. To switch clusters, edit the defaults in `.slurm_defaults`:

```yaml
CLUSTER_TAG:     "slurm-ocihsg"
SLURM_ACCOUNT:   "your_account"
ENROOT_IMAGE:    "/lustre/.../your-image.sqsh"
```

---

## Quick reference

| Task | Command |
|---|---|
| First-time setup | `python3 setup_runner.py setup` |
| Start after reboot | `python3 setup_runner.py start` |
| Stop | `python3 setup_runner.py stop` |
| Check health + logs | `python3 setup_runner.py status` |
| Live log stream | `tail -f /home/ankurv/data/ci/runner.log` |
| Full teardown | `python3 setup_runner.py cleanup --yes` |
| List registered runners | `/home/ankurv/data/ci/gitlab-runner list --config /home/ankurv/data/ci/config.toml` |
