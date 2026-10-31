#!/usr/bin/env bash
# Register GitHub Actions runners on this machine, each as its own systemd service.
#
#   add_runners.sh LABEL COUNT
#
# Prompts for a registration token from the repository's
# Settings > Actions > Runners > New self-hosted runner page.
set -Eeuo pipefail

label=${1:?usage: add_runners.sh LABEL COUNT}
count=${2:?usage: add_runners.sh LABEL COUNT}
version=2.338.0
repo_url=https://github.com/ankurv-nvidia/nvrx_checkpointing
tarball="${HOME}/actions-runner-linux-x64-${version}.tar.gz"

read -rsp "Runner registration token: " token
echo
if [[ ! -f "${tarball}" ]]; then
    curl -fsSL -o "${tarball}" \
        "https://github.com/actions/runner/releases/download/v${version}/actions-runner-linux-x64-${version}.tar.gz"
fi
for index in $(seq -w 1 "${count}"); do
    name="$(hostname)-${label}-${index}"
    dir="${HOME}/actions-runner-${label}-${index}"
    if [[ -f "${dir}/.runner" ]]; then
        echo "${name}: already registered"
        continue
    fi
    mkdir -p "${dir}"
    tar -xzf "${tarball}" -C "${dir}"
    (
        cd "${dir}"
        ./config.sh --unattended --url "${repo_url}" --token "${token}" \
            --name "${name}" --labels "${label}" --work _work
        sudo ./svc.sh install "$(id -un)"
        sudo ./svc.sh start
    )
    echo "${name}: registered and started"
done
