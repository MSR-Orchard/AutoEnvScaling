# Proposer skills

The proposer's prompt begins with two skills (Appendix "Proposer Skills" of the paper):

1. **`create-task`**, Harbor's official skill for building a valid Harbor task. It is not stored
   in this repository. The proposer image fetches `skills/create-task` from
   https://github.com/harbor-framework/harbor at commit
   `f5e9d0b71ac4493a4f0620653e2913aee7fc0767` (the revision used in the paper), together with
   Harbor's Apache-2.0 `LICENSE`; see `docker/proposer/Dockerfile`.
2. **`auto_env_scaling`** (this directory): what kind of task to build (reward spread across the
   solver's attempts), how to ground it in the solver's rollouts and past environments, and how
   to check it before submitting. `verifier_template.sh` is a fraction-scoring `tests/test.sh`.

**`harbor_domains`** holds guidance for software engineering, tool calling, browser use,
computer use, GPU kernels and professional workflows (Appendix "Cross-Domain Environment
Generation"). It is staged in the workspace but not inlined in the prompt.

At each episode the host copies the image's skills into the workspace's read-only `skills/`
directory. To try edited skills without rebuilding the image, set `AES_PROPOSER_SKILLS_DIR` to a
directory with the same layout (it must then include `create-task/` as well).
