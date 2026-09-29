# Miles

AutoEnvScaling trains policies with [Miles](https://github.com/radixark/miles), pinned at commit
`5e392cf2cef761d4508e81b2a8f4c4b191d7a73c`, plus the changes in `miles-autoenvscaling.patch`.

## Applying the patch

```bash
git clone https://github.com/radixark/miles
cd miles
git checkout 5e392cf2cef761d4508e81b2a8f4c4b191d7a73c
git apply /path/to/AutoEnvScaling/third_party/miles/miles-autoenvscaling.patch
```

Build the image from `docker/Dockerfile` in that checkout, or mount the patched checkout into
the upstream Miles image. The patch touches Python only, so no rebuild of SGLang or
Megatron-LM is needed.

## What the patch changes

- **DPPO** (`--advantage-estimator dppo`, `--dppo-divergence-type {tv,kl}`,
  `--dppo-divergence-threshold`): a trust region enforced by masking tokens whose binary
  divergence from the rollout policy exceeds the threshold and whose update would move further
  away, instead of PPO clipping. The paper uses total variation with threshold 0.1.
- **Colocated serving guards**: an SGLang engine leaves the router before its memory is paged
  out and rejoins after it is restored, so a late request from an agent cannot crash the job;
  `flush_cache` waits for straggler requests instead of failing after about a second.
- **`--rollout-server-startup-timeout-secs`**: the router start-up wait is configurable
  (upstream: fixed 30 s).
- Two small fixes: `Dataset` falls back to the unfiltered samples when `filter_long_prompt`
  skips filtering, and `process_vision_info` accepts prompts that are already strings.

DPPO is not in Miles' built-in reward normalization, so RL runs also need
`--custom-reward-post-process-path autoenvscaling.training.miles.hooks.center_rewards_by_group`,
which computes the paper's advantage (reward minus group mean, no std division). The configs in
`autoenvscaling.training.miles.config` set it.

## How AutoEnvScaling drives Miles

`python -m autoenvscaling.training.miles plan|run --preset <name>` builds the `train.py`
arguments from `MilesConfig` and runs `autoenvscaling/training/miles/launch/train.sh` inside the
Miles container (or `launch/apptainer.sh` when `MILES_IMAGE` is set). Set `MILES_DIR` and
`MEGATRON_DIR` to the checkouts inside the container. `ref_load` is the starting checkpoint in
Megatron format, produced with Miles' `tools/convert_hf_to_torch_dist.py`.

Three modes:

- **bridge** (default). The self-play loop (`python -m autoenvscaling.selfplay run`) runs every
  solver attempt, pool review and proposer episode against the Miles router, groups and scores
  the trajectories, and hands each finished batch to Miles through a shared control directory
  (`--bridge-dir` on both commands). Miles re-tokenizes the conversations with its loss-mask
  generator and trains one step per batch. Start both processes with the same directory; the
  loop's `schedule` subcommand prints the `--num-rollout` to use. Sandboxes must be able to reach
  the router: pass `--public-base-url` to the loop when they run elsewhere (for example on
  Modal) and the router is exposed through a tunnel.
- **generate_fn**. Miles samples prompts from a pool file (`Pool.export_prompt_data`) and calls
  a per-episode `--custom-generate-function-path` that runs the solver in a sandbox and returns
  a `Sample` with tokens and log-probabilities. AutoEnvScaling does not ship such a function.
- **sft**. Supervised fine-tuning on `python -m autoenvscaling.trajectories.export sft` output
  (`--input-key messages --tool-key tools`), used for the Cold-Start checkpoint.

## Known limitations

- Bridge mode trains on re-tokenized conversations. Any difference between the re-rendered
  chat template and the tokens the engine actually produced (for example reasoning text the
  template drops from earlier turns) is trained on as if it had been sampled. Exact token
  capture needs a proxy between the agent harness and the router that records token ids.
- Unfinished trajectories are not carried across updates: every episode in a batch finishes (or
  fails) before the update, so the step waits for its slowest episode.
- Evaluation does not run inside Miles in bridge mode; evaluate saved checkpoints separately.
- Resuming: restart Miles with `--load` and the loop with the same `--run-dir`. Batches already
  written for rollout ids after the checkpoint are replayed, and the loop's counters can be
  ahead of the checkpoint by up to `save_interval` updates.
