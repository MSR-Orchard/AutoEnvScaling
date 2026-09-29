"""Miles run configuration. The defaults are the paper's shared-model settings (Appendix,
"Training and Calibration Settings"):

    learning rate                  1e-6 (constant)
    solver rollout batch           16 environments x 16 trajectories
    DPPO total-variation threshold 0.1
    KL / entropy coefficients      0 / 0

`train_args()` renders the arguments for Miles' `train.py`. Three modes decide where the
rollouts come from:

* `bridge` (default): the self-play loop in this package runs every solver and proposer episode
  and hands Miles finished RL groups through `training.miles.bridge`. Miles only trains and
  serves the current weights. Used for shared-policy training, frontier-proposer runs and
  static pools alike.
* `generate_fn`: Miles samples prompts from `prompt_data` (see `Pool.export_prompt_data`) and
  calls an external per-episode generate function (`--custom-generate-function-path`) that runs
  the solver in a sandbox. This package does not ship such a function; see the README.
* `sft`: supervised fine-tuning on `to_sft` rows, used for the Cold-Start checkpoint.

Paths are never defaulted: model, data and output locations come from the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

BRIDGE = "bridge"
GENERATE_FN = "generate_fn"
SFT = "sft"
MODES = (BRIDGE, GENERATE_FN, SFT)

BRIDGE_ROLLOUT_FN = "autoenvscaling.training.miles.rollout.generate_rollout"
REWARD_HOOK = "autoenvscaling.training.miles.hooks.center_rewards_by_group"
NONZERO_STD_FILTER = "miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
SFT_ROLLOUT_FN = "miles.rollout.sft_rollout.generate_rollout"


@dataclass
class MilesConfig:
    # ---- model and checkpoints ------------------------------------------------------------
    model_args_script: str = "scripts/models/qwen3.6-35B-A3B.sh"   # relative to MILES_DIR
    hf_checkpoint: str = ""           # Hugging Face weights of the starting policy
    ref_load: str = ""                # the same weights converted to Megatron torch_dist
    load: str | None = None           # a Miles save directory to resume from
    save: str = ""                    # where Miles writes checkpoints
    save_interval: int = 16           # one checkpoint per pool refresh

    # ---- where rollouts come from ------------------------------------------------------------
    mode: str = BRIDGE
    prompt_data: str | None = None                # generate_fn and sft modes
    generate_function_path: str | None = None     # generate_fn mode
    over_sampling_batch_size: int = 24            # generate_fn mode: prompts drawn per batch

    # ---- batch and schedule --------------------------------------------------------------------
    rollout_batch_size: int = 16      # environments per solver batch
    n_samples_per_prompt: int = 16    # attempts per environment
    num_rollout: int = 305            # trainer updates; see Schedule.planned_updates
    num_epoch: int | None = None      # sft mode

    # ---- lengths -------------------------------------------------------------------------------
    seq_length: int = 131072
    rollout_max_context_len: int = 131072
    rollout_max_response_len: int = 16384   # per model call
    rollout_temperature: float = 1.0

    # ---- algorithm -----------------------------------------------------------------------------
    advantage_estimator: str = "dppo"
    dppo_divergence_type: str = "tv"
    dppo_divergence_threshold: float = 0.1
    kl_coef: float = 0.0
    kl_loss_coef: float = 0.0
    entropy_coef: float = 0.0
    loss_mask_type: str = "qwen3"

    # ---- optimizer -----------------------------------------------------------------------------
    lr: float = 1e-6
    lr_decay_style: str = "constant"
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    clip_grad: float = 1.0

    # ---- parallelism (one 8-GPU node, colocated training and inference) -------------------------
    num_nodes: int = 1
    gpus_per_node: int = 8
    tensor_parallel: int = 2
    context_parallel: int = 4
    pipeline_parallel: int = 1
    expert_parallel: int = 8
    expert_tensor_parallel: int = 1
    moe: bool = True
    linear_attention_backend: str | None = "fla"   # Qwen3.5 and Qwen3.6 hybrid attention

    # ---- serving -------------------------------------------------------------------------------
    rollout_num_gpus_per_engine: int = 8
    sglang_mem_fraction: float = 0.7
    sglang_context_length: int = 262144     # proposer episodes can exceed 200k tokens
    sglang_tool_call_parser: str = "qwen3_coder"
    sglang_reasoning_parser: str = "qwen3"
    served_model_name: str = "policy"
    server_startup_timeout_s: int = 300     # router start-up; needs the third_party/miles patch
    sglang_extra: list[str] = field(default_factory=lambda: [
        "--sglang-mamba-scheduler-strategy", "extra_buffer",
        "--sglang-ep-size", "1",
        "--sglang-moe-runner-backend", "triton",
        "--sglang-cuda-graph-bs-decode", "1", "2", "4", "8", "16", "32", "64", "128"])

    # ---- logging -------------------------------------------------------------------------------
    wandb_project: str | None = None
    wandb_group: str | None = None
    extra_args: list[str] = field(default_factory=list)

    def with_(self, **changes: Any) -> MilesConfig:
        return replace(self, **changes)

    @property
    def global_batch_size(self) -> int:
        return self.rollout_batch_size * self.n_samples_per_prompt

    def problems(self) -> list[str]:
        """Everything that would make the run fail or train something other than intended."""
        out = []
        if self.mode not in MODES:
            out.append(f"mode must be one of {MODES}, got {self.mode!r}")
        for name in ("hf_checkpoint", "save"):
            if not getattr(self, name):
                out.append(f"{name} is required")
        if self.mode != SFT and not self.ref_load:
            out.append("ref_load (the torch_dist checkpoint) is required for RL")
        if self.mode in (GENERATE_FN, SFT) and not self.prompt_data:
            out.append(f"{self.mode} mode needs prompt_data")
        if self.mode == GENERATE_FN and not self.generate_function_path:
            out.append("generate_fn mode needs generate_function_path")
        world = self.num_nodes * self.gpus_per_node
        model_parallel = self.tensor_parallel * self.context_parallel * self.pipeline_parallel
        if world % model_parallel:
            out.append(f"TP*CP*PP = {model_parallel} does not divide {world} GPUs")
        if self.sglang_context_length < self.rollout_max_context_len:
            out.append("sglang_context_length must be >= rollout_max_context_len")
        return out

    # ---- rendering -----------------------------------------------------------------------------
    def train_args(self) -> list[str]:
        """Arguments for Miles' train.py, after the model architecture arguments."""
        a: list[str] = [
            "--actor-num-nodes", str(self.num_nodes),
            "--actor-num-gpus-per-node", str(self.gpus_per_node),
            "--colocate",
            "--hf-checkpoint", self.hf_checkpoint,
            "--save", self.save, "--save-interval", str(self.save_interval),
        ]
        if self.ref_load:
            a += ["--ref-load", self.ref_load]
        if self.load:
            a += ["--load", self.load]
        a += self._mode_args()
        a += ["--optimizer", "adam", "--lr", _num(self.lr), "--lr-decay-style", self.lr_decay_style,
              "--weight-decay", _num(self.weight_decay), "--adam-beta1", _num(self.adam_beta1),
              "--adam-beta2", _num(self.adam_beta2), "--clip-grad", _num(self.clip_grad)]
        a += self._parallel_args()
        if self.mode != SFT:
            a += self._serving_args()
        a += ["--attention-dropout", "0.0", "--hidden-dropout", "0.0",
              "--accumulate-allreduce-grads-in-fp32", "--attention-softmax-in-fp32",
              "--attention-backend", "flash"]
        if self.wandb_project:
            a += ["--use-wandb", "--wandb-project", self.wandb_project]
            if self.wandb_group:
                a += ["--wandb-group", self.wandb_group]
        return a + list(self.extra_args)

    def _mode_args(self) -> list[str]:
        if self.mode == SFT:
            a = ["--rollout-function-path", SFT_ROLLOUT_FN, "--prompt-data", str(self.prompt_data),
                 "--input-key", "messages", "--tool-key", "tools", "--metadata-key", "metadata",
                 "--rollout-shuffle", "--rollout-batch-size", str(self.rollout_batch_size),
                 "--global-batch-size", str(self.rollout_batch_size),
                 "--loss-type", "sft_loss", "--calculate-per-token-loss",
                 "--disable-compute-advantages-and-returns", "--debug-train-only",
                 "--loss-mask-type", self.loss_mask_type]
            if self.num_epoch is not None:
                a += ["--num-epoch", str(self.num_epoch)]
            else:
                a += ["--num-rollout", str(self.num_rollout)]
            return a

        a = ["--rollout-batch-size", str(self.rollout_batch_size),
             "--n-samples-per-prompt", str(self.n_samples_per_prompt),
             "--global-batch-size", str(self.global_batch_size),
             "--num-steps-per-rollout", "1",
             "--num-rollout", str(self.num_rollout),
             "--rollout-max-context-len", str(self.rollout_max_context_len),
             "--rollout-max-response-len", str(self.rollout_max_response_len),
             "--rollout-temperature", _num(self.rollout_temperature),
             # DPPO with mean-centred advantages computed per group (no std division).
             "--advantage-estimator", self.advantage_estimator,
             "--dppo-divergence-type", self.dppo_divergence_type,
             "--dppo-divergence-threshold", _num(self.dppo_divergence_threshold),
             "--kl-coef", _num(self.kl_coef), "--kl-loss-coef", _num(self.kl_loss_coef),
             "--kl-loss-type", "low_var_kl", "--entropy-coef", _num(self.entropy_coef),
             "--custom-reward-post-process-path", REWARD_HOOK]
        if self.mode == BRIDGE:
            # Batches arrive already grouped and filtered; their size varies (a proposer update
            # has proposer_group_size samples per group), so each one trains as a single step.
            a += ["--rollout-function-path", BRIDGE_ROLLOUT_FN, "--disable-rollout-global-dataset",
                  "--use-dynamic-global-batch-size", "--loss-mask-type", self.loss_mask_type]
        else:
            a += ["--prompt-data", str(self.prompt_data), "--input-key", "prompt",
                  "--label-key", "label", "--metadata-key", "metadata", "--apply-chat-template",
                  "--custom-generate-function-path", str(self.generate_function_path),
                  "--over-sampling-batch-size", str(self.over_sampling_batch_size),
                  "--dynamic-sampling-filter-path", NONZERO_STD_FILTER]
        return a

    def _parallel_args(self) -> list[str]:
        a = ["--tensor-model-parallel-size", str(self.tensor_parallel), "--sequence-parallel",
             "--context-parallel-size", str(self.context_parallel),
             "--pipeline-model-parallel-size", str(self.pipeline_parallel),
             "--recompute-granularity", "full", "--recompute-method", "uniform",
             "--recompute-num-layers", "1", "--qkv-format", "thd", "--micro-batch-size", "1",
             "--seq-length", str(self.seq_length)]
        if self.moe:
            a += ["--expert-model-parallel-size", str(self.expert_parallel),
                  "--expert-tensor-parallel-size", str(self.expert_tensor_parallel),
                  "--moe-token-dispatcher-type", "flex"]
        if self.linear_attention_backend:
            a += ["--linear-attention-backend", self.linear_attention_backend]
        return a

    def _serving_args(self) -> list[str]:
        return ["--rollout-num-gpus-per-engine", str(self.rollout_num_gpus_per_engine),
                "--sglang-mem-fraction-static", _num(self.sglang_mem_fraction),
                "--sglang-context-length", str(self.sglang_context_length),
                "--sglang-tool-call-parser", self.sglang_tool_call_parser,
                "--sglang-reasoning-parser", self.sglang_reasoning_parser,
                "--sglang-served-model-name", self.served_model_name,
                "--rollout-server-startup-timeout-secs", str(self.server_startup_timeout_s),
                *self.sglang_extra]


def _num(x: float) -> str:
    return repr(x) if isinstance(x, float) else str(x)


# ---------------------------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------------------------
_DENSE_SGLANG = ["--sglang-mamba-scheduler-strategy", "extra_buffer",
                 "--sglang-cuda-graph-bs-decode", "1", "2", "4", "8", "16", "32", "64", "128"]


def shared_policy(**paths: Any) -> MilesConfig:
    """Recursive self-improvement: Qwen3.6-35B-A3B as proposer and solver, 288 solver updates
    plus 17 proposer updates (`Schedule(288).planned_updates`)."""
    return MilesConfig(**paths)


def exp2_solver(model: str = "qwen3.5-9b", **paths: Any) -> MilesConfig:
    """Solver RL with a fixed-weight proposer or a static pool (Section 3.3 experiments):
    group size 16, 16 prompts per batch, 128k maximum sequence length."""
    common = {"num_rollout": 300, "wandb_group": f"exp2-{model}"}
    if model == "qwen3.5-9b":
        cfg = MilesConfig(model_args_script="scripts/models/qwen3.5-9B.sh", tensor_parallel=4,
                          context_parallel=2, moe=False, rollout_num_gpus_per_engine=1,
                          sglang_context_length=131072, sglang_extra=list(_DENSE_SGLANG), **common)
    elif model == "qwen3.5-35b-a3b":
        cfg = MilesConfig(model_args_script="scripts/models/qwen3.5-35B-A3B.sh",
                          sglang_context_length=131072, **common)
    else:
        raise ValueError(f"unknown exp2 solver {model!r} (qwen3.5-9b or qwen3.5-35b-a3b)")
    return cfg.with_(**paths)


def exp2_qwen35_9b(**paths: Any) -> MilesConfig:
    """Solver RL for Qwen3.5-9B with a fixed-weight proposer or a static pool."""
    return exp2_solver("qwen3.5-9b", **paths)


def exp2_qwen35_35b(**paths: Any) -> MilesConfig:
    """Solver RL for Qwen3.5-35B-A3B with a fixed-weight proposer or a static pool."""
    return exp2_solver("qwen3.5-35b-a3b", **paths)


def cold_start_sft(**paths: Any) -> MilesConfig:
    """Cold-Start: fine-tune Qwen3.6-35B-A3B on the proposer trajectories from `to_sft`.

    The epoch count, learning rate and schedule are placeholders: the released Cold-Start
    checkpoint was trained with a different trainer and its hyperparameters are not recorded
    in the code this package was ported from.
    """
    return MilesConfig(mode=SFT, rollout_batch_size=16, num_epoch=4, lr=1e-5,
                       lr_decay_style="cosine", save_interval=100, **paths)


PRESETS = {"shared-policy": shared_policy,
           "exp2-qwen3.5-9b": exp2_qwen35_9b,
           "exp2-qwen3.5-35b-a3b": exp2_qwen35_35b,
           "cold-start": cold_start_sft}
