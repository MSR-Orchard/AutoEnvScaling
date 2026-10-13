# Partial rollout and over-sampling: the final run, the release, and what was restored

The paper's solver training did not wait for every episode of a batch to finish before an
update. Unfinished episodes were *parked* across the weight update (their sandbox and
conversation kept alive) and *resumed* under the new weights, keeping the token ids and
log-probabilities the engine had already sampled; batches were filled by over-sampling
environments and discarding groups whose rewards did not vary. The code release of 29/09/2026
did not have this: its Miles bridge waited for complete batches, stopped after three sampling
waves, could train on an under-filled batch, and re-tokenized conversations without the
engine's log-probabilities (`third_party/miles/README.md`, "Known limitations").

This document records where the final run's implementation was found, what it did (verified
from its code, its recorded configuration and its launcher), what the release now does
(`autoenvscaling/training/carry/`), which parts of the port are new decisions rather than
restored behaviour, and what could not be verified.

## 1. Evidence: the final run

| | |
| --- | --- |
| Run | W&B `spare-terminal-miles/qwen36_full_run_0913_202701`, started 2026-09-13 20:27:45 UTC; resumed from iteration 77 as `qwen36_resume77_20260916` |
| Entry point | `python3 -m train_spare_slime <args>` submitted as a Ray job by `spare-terminal-carry/tools/submit_qwen36_training.py` (saved 2026-09-13 00:23 UTC); `train_spare_slime.main` is Miles' `train()` with the SPARE arguments added |
| Source snapshot | `PYTHONPATH=/root/Megatron-LM:…/code/miles:…/code/spare-terminal-carry:…/dune_deps`; the carry modules `spare-terminal-carry/spare/miles/{terminal_generate,episode_registry,pi_rollout,pi_driver,miles_adapter,carry_metrics}.py` were last modified 2026-09-13 17:12 to 20:20 UTC, minutes before launch. The `spare-pi-dune-publish-20260914` copy is byte-identical for these files except `pi_driver.py`, `terminal_driver.py` and `aks_http_backend.py`, edited after launch |
| Miles | upstream `5e392cf` (the release's pin) with 13 locally modified files. The release patch already carries DPPO, the colocated-serving guards, `--rollout-server-startup-timeout-secs` and two small fixes. **Not** in the release patch: `miles/rollout/sglang_rollout.py` (+326 lines: the carry hooks, episode-level admission, bounded abort drain), `miles/ray/rollout/rollout_manager.py` (shutdown hook), `miles/ray/rollout/train_data_conversion.py` (DPPO in Miles' own group normalization), `miles/utils/http_utils.py` (HTTP read timeouts, file-descriptor limit) |
| Proxy | `pi_logprob_proxy.py` (935 lines) in front of the router: records SGLang's `output_token_logprobs` per episode and holds Pi's in-flight request across the weight swap |

Recorded configuration (W&B config dump and the 130 launch arguments):

| Argument | Final run |
| --- | --- |
| `--rollout-batch-size` x `--n-samples-per-prompt` | 16 x 16 |
| `--over-sampling-batch-size` | 40 |
| `--global-batch-size` | 256 (one optimizer step per rollout) |
| `--partial-rollout` | on; `--mask-offpolicy-in-partial-rollout` off |
| `--max-weight-staleness` | 4 |
| `--use-rollout-logprobs` | on; `--recompute-logprobs-via-prefill` off |
| `--dynamic-sampling-filter-path` | `check_reward_nonzero_std` |
| `--rollout-shuffle`, `--num-rollout` | on, 200 |
| `--advantage-estimator` | `dppo`, tv, 0.1; `--disable-grpo-std-normalization` |
| `--rollout-max-context-len`, `--rollout-max-response-len`, `--seq-length` | 98304, 16384, 131072 |
| `--spare-terminal-episode-timeout` / setup / verify | 2700 s / 900 s / 900 s, 128 turns |
| `--custom-generate-function-path` | `spare.miles.miles_adapter.generate` (driver `spare.miles.pi_driver.make_driver`) |
| `--custom-rollout-log-function-path` | `spare.miles.carry_metrics.log_rollout_data` |
| Sandboxes | `DUNE_MAX_LIVE_SANDBOXES=1024`, `SPARE_MAX_CONCURRENT_EPISODES=768`, `SPARE_CARRY_ALLOWANCE=768` on the resume (see section 5) |

## 2. The release before this change

| | Release bridge (`training/miles/{bridge,rollout}.py`, `selfplay/loop.py`) | Final run |
| --- | --- | --- |
| Who runs episodes | the self-play loop, through Harbor trials | Miles' rollout manager, one sandbox per episode |
| Batch filling | `need + 8` environments per wave, at most 3 waves, then train on whatever passed the filter (possibly fewer than 16 groups, possibly none) | draw 40 environments per refill against an episode budget, keep going until exactly 16 groups pass the filter |
| Unfinished episodes | every episode finishes before the update | parked at the cut, resumed under the new weights |
| Training tokens | the conversation re-tokenized with the chat template; behaviour log-probabilities recomputed by the actor | the engine's sampled ids and log-probabilities, recorded by the proxy, never recomputed |
| Staleness | not applicable | parked episodes may span up to 4 updates, measured from admission |

## 3. Verified original behaviour, now in the release

Each item names the research file it was taken from and the release file that holds it.

**Episodes survive the update** (`terminal_generate.py`, `episode_registry.py` -> `carry/episode.py`,
`carry/registry.py`). At the cut the rollout's abort flag flips; an episode that has not finished
returns `ABORTED`, its sandbox and turn state are stored in the registry under the episode id
(`group-<group_index>-r<index>`, stable across park and resume), and its sample keeps the
tokens, `rollout_log_probs` and `loss_mask` sampled so far plus `metadata.admission_version`,
`start_rollout_id` and `episode_id`. When Miles hands the sample back from its buffer, the
episode resumes in the same sandbox: the registry records a `ResumeBoundary` (token index,
turn, parked-at and resumed-at versions), the sandbox's clock resumes, and new tokens are
appended by the new policy. On completion the sample carries `metadata.resumes` and
`resume_boundaries`. Behaviour log-probabilities are never recomputed, which is why
`--use-rollout-logprobs` is required and `--mask-offpolicy-in-partial-rollout` and
`--recompute-logprobs-via-prefill` are refused (`carry/rollout.py::config_problems`).

**A miss restarts the episode** (`terminal_generate.py` lines 154-173). A sample with a partial
response but no registry entry (after a process restart, or after a stale eviction) has its
partial tokens discarded and starts from turn 0: it costs work, never correctness.

**Staleness is stamped at admission and bounded by the carry** (`pi_rollout.py::_stale`,
`group_error`; `episode_registry.py::evict_stale`). Miles only enforces `--max-weight-staleness`
in its fully asynchronous buffer, so the carry does it: a group whose admission version is more
than 4 updates behind is discarded whole, and the registry evicts per episode (not per task).

**Groups are atomic** (`pi_rollout.py::group_error`, `discard_group`, `reconcile_buffer`,
`take_buffered_groups` -> `carry/lifecycle.py`). Before training, every member must be completed
with consistent tokens, log-probabilities and mask (`invalid_token_lengths`,
`invalid_logprobs_or_mask`, `invalid_reward`); before a resume, every unfinished member must
still own a parked sandbox (`missing_sandbox`, `episode_already_active`, `timeout`). A failing
member discards the group (`episode_failed`) and evicts its sandboxes. After the cut the buffer
is reconciled at the next version and every sandbox not reachable from it is released.

**Over-sampling and the filter** (the `sglang_rollout.py` patch -> `carry/rollout.py::collect_batch`).
The buffer is claimed first, whole. Fresh groups are drawn `over_sampling_batch_size` at a time
and admitted against `deficit = max_concurrent_episodes - in_flight - held`, counted in episodes
(in-flight is decremented per episode, not per group). A finished group passes
`check_reward_nonzero_std` or is dropped and its sandboxes released; the loop continues until
`rollout_batch_size` groups passed, with no limit on draws. A batch is therefore never
under-filled, and an exhausted data source raises instead of training short. Groups that
complete after the batch is full are carried to the next rollout as completed groups, where
they are re-checked (staleness included) and reused without regeneration.

**The cut** (`sglang_rollout.py::abort` patch, `pi_rollout.py::before_cut`, `after_engine_abort`
-> `carry/rollout.py::cut`, `carry/lifecycle.py::after_engine_abort`). The proxy's global gate
closes, the engines receive `abort_request abort_all`, abort sweeps repeat until the proxy
confirms every engine call has settled (fail-closed: a timeout never authorises training), and
pending group tasks are drained with a bound (300 s) after which stragglers are cancelled and
restart next rollout. Partial groups get `start_rollout_id` and go to Miles' buffer.

**The recording proxy** (`pi_logprob_proxy.py` -> `carry/proxy.py`). Pi speaks
`/v1/ep/<episode_id>/chat/completions`; the proxy renders with the policy's chat template
(thinking preserved, tool arguments as dicts), calls `/generate` with `return_logprob`, records
ids, log-probabilities and a loss mask (prompt and tool results masked out), and parses tool
calls with SGLang's parser. Every render must extend the recorded prefix; a shorter or diverging
re-render after a park rolls back the provisional generation, a divergence before it is a 409.
The context budget and turn limit end the episode with `finish_reason: length` rather than
producing tokens the trainer cannot consume. Streaming responses send headers at once and
keepalive comments while a request is held.

**The park model** (`pi_driver.py` -> `carry/driver.py`). Pi is never killed: the driver closes
the episode's gate at the proxy and Pi blocks inside its in-flight HTTP call; a resume reopens
the gate and rejoins the running process. Active time and paused time are accounted separately
(2700 s active, 7200 s paused backstop). Verification is the terminal call and always releases
the sandbox; an episode with no recording, a Pi error, or no tool result is a failure, not a
reward.

**Sample repairs before Miles sees them** (`miles_adapter.py` -> `carry/episode.py::normalize_sample`).
A missing reward becomes 0.0; a log-probability vector that does not match `response_length`
voids the response (the sample stays inert in its group); a sample must keep one prompt token.
Each of these prevented a whole-step failure inside Megatron in the research runs.

**Capacity invariant** (`submit_qwen36_training.py`, `pi_rollout.py::begin_rollout` -> `CarrySettings`).
`n_samples_per_prompt <= max_concurrent_episodes <= carry_allowance <= max_live_sandboxes`;
admission of a sandbox never blocks a worker thread.

## 4. New in this port (design decisions, not restored behaviour)

* **A rollout function instead of a patch to `sglang_rollout.py`.** The final run's loop lived
  inside Miles (`_terminal_hooks` in the local patch). The release implements the same loop as
  `--rollout-function-path autoenvscaling.training.carry.rollout.generate_rollout`, the hook
  bridge mode already uses, with the Miles-free core `collect_batch` tested offline. The patch
  hunks were reproduced in behaviour (admission in episodes, buffer-first, bounded drain,
  cancel-on-error, surplus completed groups carried, final-batch validation); `rollout_manager`'s
  shutdown hook became the last-rollout drain plus an `atexit` backstop; the HTTP timeout hunk
  is unnecessary because the proxy, not Miles' client, calls `/generate`.
* **Sandboxes are the release's `autoenvscaling.sandbox` backends** started from the environment
  image named in the prompt row (`metadata["image"]`, with `task_dir` for the verifier), not the
  sandbox service of the research cluster. Their settings are the repository's own solver
  settings unless `AES_CARRY_*` overrides them: the backend from `Settings.sandbox.backend`
  (Modal by default), `solver_cpu` (4) and `solver_memory_gib` (8), the Terminal-Bench working
  directory `/app`, and the solver network mode (`AES_SOLVER_NETWORK`, allowlist by default).
  Under the allowlist a sandbox may reach only the proxy's host, the hosts Pi installs from
  (`PI_INSTALL_HOSTS`) and whatever the task's `task.toml` allows, exactly as the Harbor trials
  of the calibration path; the Modal backend enforces it (`SandboxSpec.domain_allowlist`, which
  the backend previously ignored), the Docker backend cannot and says so. The research run used
  2 CPUs per sandbox; the release keeps the repository's 4. A task that sets `[agent] user` is
  logged: the sandbox interface runs commands as the image's default user.
* **Run identity and orphan removal.** A sandbox is named `<run_id>-<instance_id>-<episode>`:
  the run id defaults to the Miles save directory (one training job, shared by its restarts,
  never by two jobs; `AES_CARRY_RUN_ID` overrides) and the instance id is one rollout process.
  Each process writes a liveness lease under `<state_dir>/instances/` before its first sandbox
  and refreshes it from the heartbeat thread. On start a process removes only the sandboxes of
  instances of its own run whose lease is stale (`AES_CARRY_LEASE_STALE_S`, 900 s): a live
  instance, another run, and sandboxes with no lease are never touched, and without a state
  directory nothing is reaped. Docker removes by container-name prefix, Modal by the `aes.name`
  tag (untested against the live SDK, see section 5).
* **Capacity accounting.** The rollout's admission deficit is `capacity - in_flight - held`,
  where `held` counts only sandboxes whose deletion has started but is not confirmed; active
  sandboxes are counted once, by the in-flight counter (or by the registry while parked).
* **Pi is installed as Harbor installs it**: `@earendil-works/pi-coding-agent@0.80.6`
  (`calibration.solver.PI_VERSION`; the old `@mariozechner` scope stopped at 0.73.1 and is
  deprecated on npm, so the research launcher's package name no longer resolves), with
  `npm install -g --ignore-scripts` after Harbor's nvm/Node 22 snippet, and its endpoint
  configuration in `PI_CODING_AGENT_DIR` rather than the home directory. `AES_CARRY_PI_PACKAGE`
  and `AES_CARRY_PI_VERSION` override both.
* **Verification is the task's `tests/test.sh`** writing `/logs/verifier/reward.{txt,json}`
  (the convention every other check in this repository uses), rather than a pytest source
  embedded in the research artifact. `/tests` and `/logs/verifier` are removed and recreated
  after Pi has exited and before the tests are uploaded, so a reward file planted by the agent
  or left by an earlier attempt can never score the episode. The verifier contract is explicit
  (`driver.py::parse_reward_output`): Harbor ignores the test script's exit status and lets the
  reward file decide, and the admitted tasks were validated under that rule, so exit 0 and
  exit 1 (ordinary test failure, pytest's code) are accepted together with a well-formed reward
  in [0, 1]; any other exit status (a crashed, timed-out or incomplete verifier), a missing or
  malformed reward file, or exit 1 reporting a full reward is a verifier failure: the episode is
  dropped and never becomes a training sample. The research driver also ran an initial-state
  check before the agent; the release's tasks have no such check and none is run.
* **The proxy authenticates every endpoint** with `AES_PROXY_API_KEY` (only `/healthz` is
  open): the control surface can change the engine address, open the gates and read or delete
  recordings, and the research proxy only protected completions. The driver and the rollout's
  gate client send the same key. A render that is shorter than the committed recording even
  after the provisional rollback is refused (409) instead of being generated from, and the
  initial `abort_request` at the cut is bounded (`AES_CARRY_ABORT_REQUEST_TIMEOUT_S`, 60 s)
  so a silent router cannot hold the cut forever; quiescence is still confirmed or refused.
* **Settings travel as environment variables** (`AES_CARRY_*`, `AES_PROXY_*`; forwarded to the
  Ray workers by the launcher) instead of `--spare-*` arguments, because the release launches
  Miles' own `train.py`.
* **Explicit checks the research code only documented**: `--use-rollout-logprobs` must be on
  (the run had it; its code asserted the two related flags but not this one).
* **Per-generation engine weight versions** are recorded by the proxy and copied to
  `sample.weight_versions`, so Miles' own mixed-version metrics apply to carried samples.
* **Restart manifest** (`carry_manifest_<rollout>.json` under `AES_CARRY_STATE_DIR`): see
  section 7.
* **Not ported**: `group_accounting.py` (survivor floor, minimum trainable groups per cut). It
  exists in the research code but is not on the final run's path; only its metric names are
  imported there.

## 5. Not verified, or missing evidence

* **No live run.** Everything was tested offline with fakes for Miles' data source and
  arguments, the engines, the proxy's tokenizer and tool parser, and the sandboxes. The Miles
  wiring (`generate_rollout` with `RolloutDataSourceWithBuffer`, the router's worker list and
  `abort_request`), the Pi install on the release's images, SGLang's tool parser and the Modal
  backend need a paid run to confirm.
* **`SPARE_CARRY_ALLOWANCE` of the 13 September launch.** The launcher saved that day sets it to
  256 (`rollout_batch_size x n_samples_per_prompt`), which the final `pi_rollout.py::begin_rollout`
  rejects (`inflight <= carry_allowance` with inflight 768). The resume raised it to 768 with the
  note that the historical value "fails before the first rollout". Whether the original process
  ran with an environment override or an earlier check is not recorded. The release keeps the
  invariant and defaults to 768.
* **Which runs used the carry.** The W&B group of the final run is `terminal-fullft-carry` on a
  static pool (`tmax_pool_15k.jsonl`); the launcher comment says `global-batch-size stays 256 --
  we still TRAIN 16 groups`. Whether the paper's shared-policy self-play runs trained the solver
  through this path or through the bridge was not established from this machine; the release
  offers both (`carry` mode for solver RL on a pool export, `bridge` mode for the self-play
  loop's proposer updates).
* **Throughput numbers** in the launcher comments (over-sampling 24 -> 40 walk, drop rate 2.06x
  mean) are quoted there from monitoring, not re-measured here.
* **Modal orphan removal** uses `modal.Sandbox.list` and sandbox tags; the Modal SDK is not
  installed on the machine this was written on, so that call path is covered only by the test
  fakes. A listing failure is logged and leaves the orphans in place.

## 6. 16 x 16 (paper) versus 8 x 32 (older carry launcher)

The older launcher, `tools/submit_carry_training.py` (saved 2026-09-01), documents "full
geometry: rollout-batch 8 x n-samples 32" in its header. That was the geometry of the earlier
RSI-35 runs, matching upstream TMax (8 prompts x 32 samples). On 2026-08-29 it was changed to
16 x 16 with the recorded reason that the zero-variance filter was dropping 7 to 8 of every 12
groups even with 32 samples each: outcomes were near-deterministic per task, so more distinct
tasks helped the filter and more samples per task did not. Both launchers' *argument lines* set
16 x 16 (the 8 x 32 text in the header and comments is stale), 8 x 32 was briefly restored around
2026-09-07/08 for a fast-reproduction size and reverted, and the final run's recorded
configuration is 16 x 16 with over-sampling 40 and a global batch of 256. The paper's 16
environments x 16 attempts is therefore the final experiment; the release's `carry-solver`
preset uses it, and `--set rollout_batch_size=8 n_samples_per_prompt=32` reproduces the older
geometry if needed.

## 7. Interruption and restart

What Miles does: `RolloutDataSource.save(rollout_id)` writes only the prompt offset, epoch and
sample counters; the carry buffer (`RolloutDataSourceWithBuffer.buffer`) is not saved, and the
episode registry lives in the rollout worker's memory. After `--load`, the data source resumes
at the checkpointed offset.

What that means, made explicit in the release:

* **Nothing is duplicated.** Prompts drawn after the last checkpoint and before the interruption
  were never trained on; they are drawn again at the same offset and run as fresh episodes.
* **Carried work is lost, not silently.** On the first rollout of a process the carry reads the
  previous rollout's manifest (`carry_manifest_<id>.json`: buffered groups and parked episode
  ids) and logs how many groups and episodes restart from turn 0. The buffer is empty, so no
  resume is attempted against a sandbox the process does not own.
* **Orphaned sandboxes of a dead process are removed** before any new episode starts, so they
  cannot hold admission slots or be mistaken for resumable work; the process must be dead (its
  lease stale), so a concurrent process of the same run keeps its sandboxes. The proxy's
  recordings of a previous process are released as episodes are dropped.
* **Within a process**, a failing rollout fails closed: the gate is parked, running episodes
  are cancelled and every sandbox is released before the error propagates.

Carrying parked episodes *across* a process restart (persisting registry entries and
re-attaching to containers whose Pi is still blocked at the proxy) is possible with the Docker
backend but was not part of the original implementation and is not attempted.

## 8. Running it

```
# the recording proxy, in the Miles image (tokenizer of the served model, SGLang's parser)
AES_PROXY_HF_DIR=/path/to/Qwen3.6-35B-A3B AES_PROXY_API_KEY=... AES_PROXY_PORT=30000 \
  python -m autoenvscaling.training.carry proxy

# pool export and the Miles job
python -m autoenvscaling.selfplay export --pool-dir Q --out pool.jsonl     # adds image/task_dir per row
AES_CARRY_PROXY_PUBLIC_URL=https://<tunnel to the proxy> AES_CARRY_SANDBOX_BACKEND=docker \
AES_CARRY_MAX_LIVE_SANDBOXES=64 AES_CARRY_MAX_CONCURRENT_EPISODES=48 AES_CARRY_CARRY_ALLOWANCE=48 \
  python -m autoenvscaling.training.miles plan --preset carry-solver \
    --hf-checkpoint ... --ref-load ... --save ... --prompt-data pool.jsonl
```

`python -m autoenvscaling.training.carry check` validates the `AES_CARRY_*` settings offline.
Set `AES_PROXY_API_KEY` identically for the proxy, the Miles job (the launcher forwards it to
the rollout workers) and the sandboxes (Pi receives it as `OPENAI_API_KEY`); every proxy
endpoint except `/healthz` requires it.
The prompt rows need `metadata.image` (a built image of the environment) and
`metadata.task_dir`; `Pool.export_prompt_data` writes `task_dir`, and the image is the
environment's build (`metadata["image"]`), produced by the user's image pipeline.

Offline tests: `tests/test_training_carry.py` (registry, carry across updates with token and
log-probability preservation, restart miss, staleness, group validity, buffer reconciliation,
batch filling and filtering with the cut, drain timeout, exhausted source, manifests,
configuration), `tests/test_training_carry_proxy.py` (recording across turns and parks,
rollback, gates, limits, the HTTP surface), `tests/test_training_carry_driver.py` (install,
launch, park and rejoin, verification, release, orphan reaping).
