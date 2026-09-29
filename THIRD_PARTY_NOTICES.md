# Third-party notices

AutoEnvScaling is licensed under the Apache License 2.0 (`LICENSE`). It includes or fetches the
components below; preserve their notices and licence files when redistributing modified copies.

## Terminal-Bench task-implementation rubric

- Upstream: <https://github.com/harbor-framework/terminal-bench>, `docs/prompts/task-implementation.toml`
  at commit `ad8b8447d82d23456a8b99aa0b3c65e7abf0646c`
- License: Apache License 2.0
- Included as `rubrics/terminal-bench.toml`, with the `binary_reward` criterion removed (see the
  file header).

## Harbor `create-task` skill

- Upstream: <https://github.com/harbor-framework/harbor>, `skills/create-task`
- License: Apache License 2.0
- Not included in this repository: the proposer image (`docker/proposer/Dockerfile`) fetches it at
  a pinned commit together with Harbor's `LICENSE`.

## Miles

- Upstream: <https://github.com/radixark/miles>, base commit `5e392cf`
- License: Apache License 2.0
- Not included: `third_party/miles/miles-autoenvscaling.patch` is a patch against that commit.

## HiL-Bench

- Upstream: <https://github.com/hilbenchauthors/hil-bench>
- The `ask_human` server is taken from a local checkout of the upstream repository and hash-pinned;
  it is not included here. Check the upstream licence before redistributing it.

## Data and external dependencies

- Tmax tasks are converted from [allenai/TMax-15K](https://huggingface.co/datasets/allenai/TMax-15K)
  (ODC-BY); the dataset is not included.
- Terminal-Bench 2.1 and 4.0 tasks are fetched from the Harbor registry at run time and are not
  included; their own licences and terms apply.
- Harbor, Modal, Claude Code, Codex, Pi, PyTorch, Megatron-LM, SGLang, and model checkpoints are
  external dependencies under their own licences and terms.
