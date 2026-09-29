# GPU kernels

The task asks the solver to write a GPU kernel that is both correct and fast enough, and your
verifier establishes both against a reference implementation you also write. It is the terminal
format plus an accelerator: the same Harbor package and the same admission checks, with one
extra resource.

## What the host provides

An offered image pins PyTorch and CUDA by digest, and the host attaches the GPU as one fixed
device. Never name a device, a device index or a CDI reference in the task; a task that sets
`CUDA_VISIBLE_DEVICES` breaks on a host with a different topology.

## Write a Dockerfile on top of the offered image

Write `environment/Dockerfile` starting `FROM` the offered image by digest, and do not set
`environment.docker_image`. Add what the stock image lacks, typically an unprivileged solver
account and compilation caches inside the workspace:

```dockerfile
FROM <the offered digest>
USER root
RUN groupadd --gid 1000 user \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash user \
    && install -d -o user -g user -m 0755 /workspace
# Some hosts run /usr/bin/python3; make sure it is the image's interpreter.
RUN ln -sf "$(command -v python)" /usr/bin/python3
# Caches inside /workspace disappear with the container.
ENV TRITON_CACHE_DIR=/workspace/.triton \
    TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions \
    CUDA_CACHE_PATH=/workspace/.cache/cuda
WORKDIR /workspace
```

Set cache variables with `ENV` in the Dockerfile, not in `[environment.env]`, which resolves
`${VAR}` from the host. A cache outside `/workspace` survives into the next check, and a kernel
that is "correct" only because a previous trial's compiled artifact is still on disk fails the
reset check.

## Declare the accelerator

```toml
[environment]
gpus = 1
gpu_types = ["<the GPU type your assignment names>"]
network_mode = "no-network"
```

Ask for exactly the GPUs you use. Harbor refuses to start a task whose GPU request exceeds what
the environment offers.

## Shape of the task

Ship the reference implementation as a public, read-only input (conventionally `problem.py`,
installed at mode 0444). It defines the computation in plain PyTorch and is the ground truth for
correctness and the performance baseline. The solver writes its own implementation; your
`solution/solve.sh` installs the reference answer kernel. Never put the answer kernel anywhere
the solver can read it: `solution/` is private, public assets are not.

## The verifier runs the solver's code, so check it first

Grading a kernel means executing code the solver wrote inside the verifier. Before importing or
running it, parse it and refuse it if it does more than compute: allow only a short list of
imports (`torch`, `triton`, `math`, `typing`); refuse `exec`, `eval`, `compile`, `__import__`,
`open`, `os`, `sys`, `subprocess`, `ctypes`, `socket`, `importlib`, `pickle` and `shutil`; refuse
dunder attribute access other than `__init__`; and refuse string literals naming `/logs`,
`/tests`, `/solution` or the reward file.

## Grading

Award 1 only when the candidate compiles, is numerically correct against the reference over
several randomized inputs (not one fixed tensor), and runs within the speed ratio stated in the
instruction, measured over enough trials to be stable (twenty is a reasonable default) against
the reference timed the same way in the same process. State the ratio in the instruction as a
number the solver can check, and reject an implausible speedup as well as a slow one: a result
far faster than the reference usually means the kernel skips part of the problem.

Synchronize the device before and after every timed region, discard warm-up iterations, and
report the mean and standard deviation so that a marginal result is visible.

## Resources and checks

Give the build time to compile extensions and the verifier time for warm-up plus trials, within
the assignment's limits. The reference solution must score 1, a do-nothing agent 0 (so do not
leave a working implementation where the verifier reads), and a reset run on a fresh instance
must score 1 again without relying on a warm cache.
