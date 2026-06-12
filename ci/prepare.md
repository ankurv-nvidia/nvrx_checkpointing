# CI Runner Setup on Cluster Login Nodes

All state — token, config, binary, logs — lives under `/home/ankurv/data/ci`.
Lock down access once: `chmod 700 /home/ankurv/data/ci`

Script: `megatron/native/ci/runner/setup_runner.py`

---

## Directory layout

```
/home/ankurv/data/ci/
does th ├── token             ← registration token (you write this, chmod 600)
├── config            ← settings: GITLAB_URL, PLATFORM, CLUSTER, CONCURRENT, GPUS_PER_NODE
├── srun.tmpl            ← (optional) per-cluster override of the job-launch srun template
├── enroot_import.tmpl   ← (optional) per-cluster override of the base-image import srun command
├── image_prep.tmpl      ← (optional) per-cluster override of the project-image build srun command
├── gitlab-runner     ← binary (downloaded by setup)
├── config.toml       ← runner config (written by setup after registration)
├── runner.pid        ← PID of running process
├── runner.log        ← runner log output
├── builds/           ← job build directories
└── cache/            ← job cache
```

---

## 1. Get a registration token

1. Go to **https://gitlab-master.nvidia.com/ankurv/\<project\>/-/settings/ci_cd**
2. Expand **Runners → New project runner**
3. Set a description (e.g. `gb200 on cwdfw`) and the tag that routes jobs to this runner — the **PLATFORM**, e.g. `gb200` / `h100` / `h100` / `devbox` — then click **Create runner**
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

# Write settings
cat > $CI_DIR/config <<EOF
GITLAB_URL=https://gitlab-master.nvidia.com
PLATFORM=gb200       # required — hardware platform: GitLab runner tag + baseline namespace (gb200/h100/h100/devbox)
CLUSTER=cwdfw        # required — datacenter that selects the srun templates (cwdfw/ocihsg/...)
CONCURRENT=4
GPUS_PER_NODE=4      # GPUs per node for srun jobs on this cluster (gb200=4, h100=8, devbox=2)
EOF
chmod 600 $CI_DIR/config
```

`PLATFORM`, `CLUSTER`, `GPUS_PER_NODE` and `SLURM_PARTITION` live here (per runner)
rather than in the CI YAML, so the pipeline stays cluster-agnostic — no `--gpus-per-node`
and no `--slurm` flag in `JOB_ARGS`. `setup_runner.py` forwards them (and `CI_DIR`) into
the runner environment, which `run_job.py` honors. `PLATFORM` (gb200/h100/h100/devbox)
is the hardware identity: it matches the GitLab runner tag that routes jobs here and
namespaces the baseline files (`<PLATFORM>_<job>_<config>.yaml`). `CLUSTER`
(cwdfw/ocihsg/...) is the datacenter: it selects the per-cluster srun/enroot_import/image_prep
templates. `SLURM_ACCOUNT` being set is what marks a cluster SLURM-based — a runner
without it (e.g. devbox) runs jobs inline.

Each of these settings can equivalently be passed on the command line to
`setup`/`start` — `--cluster`, `--gitlab-url`, `--concurrent`, `--gpus-per-node`,
`--slurm-partition`. A flag **overrides** the config file for that run and
is **written back** to it, so it becomes the default on the next run (CLI wins,
then caches). For example:

```bash
python3 setup_runner.py setup --platform gb200 --cluster cwdfw --gpus-per-node 4 --slurm-account my_account --slurm-partition batch
# later, no flags needed — the cached config is reused:
python3 setup_runner.py start
```

Each workload job builds the enroot image in its **`before_script`**
(`prepare_image.py`): it creates the `.sqsh` if missing and is a fast no-op
otherwise (and a no-op on inline/devbox runners). Building it in the workload
job — rather than a separate prepare stage other jobs `needs:` — keeps the build
on the same machine that runs the job, so a runner tag spanning multiple machines
can't route the build and the workload to different hosts and leave the image
missing. Each job builds into its own per-job path (under `$CI_PROJECT_DIR`) and
moves the result to the shared image path only if that path is still absent — so
concurrent jobs on a cold machine never trample a shared scratch file, and the
first to finish populates the shared image the rest reuse. Duplicate builds are
possible but rare, since the shared image is normally already present.

### Customizing SLURM commands per cluster (template files)

The SLURM commands run_job.py issues come from template files. The repo ships
defaults that model **cw-dfw** under `megatron/native/ci/templates/`:

| Template | What it builds |
|---|---|
| `srun.tmpl` | the launch `srun` shared by every job (pretrain, ckptbench, valgrind) |
| `enroot_import.tmpl` | the `srun` that imports the base PyTorch image from the registry |
| `image_prep.tmpl` | the `srun` that runs `prepare.sh` inside the base image and saves the result |

To adapt a cluster that differs in account, partition, GPU-request syntax,
mounts, or extra srun flags, **copy the template next to this config and edit
it** — no environment variables involved:

```bash
cp <megatron-root>/megatron/native/ci/templates/srun.tmpl $CI_DIR/srun.tmpl
# then edit $CI_DIR/srun.tmpl, e.g. add `--partition=...`, replace
# `--gpus-per-node=${gpus_per_node}` with `--gres=gpu:${gpus_per_node}`,
# change `--container-mounts=...`, drop the GPU lines for clusters that
# allocate GPUs implicitly, etc.
```

If the file is **not** present in `$CI_DIR`, the repo default is used. run_job.py
substitutes the `${...}` placeholders (per-run values like `${nodes}`,
`${ntasks}`, `${gpus_per_node}`, `${job_name}`, `${image}`, `${container_env}`,
`${home}`); everything else is written literally in the file. (`setup_runner.py`
forwards `CI_DIR` so the job knows where to look.)

For a second cluster (run this on that cluster's login node with its own token):
```bash
echo 'glrt-yyyyyyyyyyyyyyyyyyyy' > /home/ankurv/data/ci/token
chmod 600 /home/ankurv/data/ci/token
# Update PLATFORM= and CLUSTER= in /home/ankurv/data/ci/config accordingly
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
