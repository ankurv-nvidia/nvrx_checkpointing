ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}

ARG NVRX_COMPUTE_CAPABILITY
ENV DEBIAN_FRONTEND=noninteractive \
    MEGATRON_ROOT=/opt/megatron-lm \
    NVRX_MEGATRON_DEPS_ROOT=/opt/nvrx-megatron-source-deps \
    MAMBA_FORCE_BUILD=TRUE \
    NVRX_RESILIENCY_EXT_VERSION=0.7.0 \
    NVRX_DEEPEP_ROOT=/opt/deepep-src \
    NVRX_COMPUTE_CAPABILITY=${NVRX_COMPUTE_CAPABILITY} \
    NVRX_RDMA_PREREQS_READY=1 \
    NVRX_USE_IMAGE_FLASH_ATTN=1

WORKDIR /workspace/nvrx_checkpointing

# The build runner's toolchain, plus the files DeepEP compiles against.
COPY scripts/prepare.sh /workspace/nvrx_checkpointing/scripts/
RUN bash scripts/prepare.sh ubuntu2404 deepep_prereqs \
 && rm -rf /var/lib/apt/lists/*

# Runtime policy changes leave the toolchain cache intact.
ENV NVRX_MEGATRON_EXTRAS=training,dev,otel,te,ssm \
    NVRX_MEGATRON_CONSTRAINTS=/workspace/nvrx_checkpointing/ci/jet/image/megatron-runtime-constraints.txt

# Native performance workloads require NVRX to be absent from the base image.
RUN python3 -m pip uninstall --yes --break-system-packages nvidia-resiliency-ext \
 && python3 -c 'import importlib.metadata as m; assert not any(d.metadata["Name"] == "nvidia-resiliency-ext" for d in m.distributions())'

# Images own the pinned Megatron source; native allocations build architecture-sensitive dependencies.
RUN bash scripts/prepare.sh megatron_root "${MEGATRON_ROOT}"

# Embed the pinned Megatron dependency sources and their submodules.
COPY ci/jet/image/install_megatron_source_deps.sh /workspace/nvrx_checkpointing/ci/jet/image/
RUN bash ci/jet/image/install_megatron_source_deps.sh \
 && git -C "${MEGATRON_ROOT}" checkout --quiet -- pyproject.toml

# Embed pinned DeepEP sources for allocation-local compilation.
COPY ci/jet/image/install_deepep_source.sh /workspace/nvrx_checkpointing/ci/jet/image/
RUN bash ci/jet/image/install_deepep_source.sh

ARG NVRX_SOURCE_REVISION
ENV NVRX_SOURCE_REVISION=${NVRX_SOURCE_REVISION}

COPY . /workspace/nvrx_checkpointing

CMD ["/bin/bash"]
