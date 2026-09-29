---
name: harbor_domains
description: Guidance for authoring complete Harbor tasks outside the terminal domain (software
  engineering, tool calling, browser use, computer use, GPU kernels, professional workflows),
  and for repairing them from the host's admission feedback. Use with create-task when the
  assignment names a domain other than terminal.
---

# Authoring Harbor tasks across domains

Use the `create-task` skill for Harbor mechanics. This overlay adds what changes when the
assignment's domain is not a plain terminal task; it does not define another task format. The
host's final instructions take precedence over interactive examples in the upstream skill.

Read the solver-failure evidence and earlier attempts before authoring, identify the failure you
target, and keep measured solver feedback separate from your own examples. If no solver feedback
is supplied, say that the attempt is a cold start; do not invent calibration results.

Scaffold with the Harbor CLI and return one complete task under `output/tasks/<name>/`,
including the instruction, `task.toml`, the environment, the reference solution and the
verifier. For a single-step Linux task the reference solution is `solution/solve.sh` and the
verifier is `tests/test.sh`. These scripts may apply patches, drive a browser or tool client, or
call native evaluators; they are not limited to terminal workflows. A Dockerfile is needed only
when the runtime builds one; tasks on a prebuilt image can use Harbor's `docker_image` field.
No later host step fills in missing task content.

Read the assignment's domain, the solver interfaces it permits, the images and resource limits
it allows, and the files it protects. Keep any runtime files the assignment supplies unchanged
in the output package. Generate meaningful setup, public assets, instruction, reference solution
and verifier. Do not treat a reference task's exact application state or answer as a limit on
what you may build.

Your shell access while authoring is not the solver's interface. Stay in the assigned domain:
incidental screenshots or clicks around a shell answer do not make a GUI workflow. A mixed
professional task may legitimately allow both command-line and GUI work.

- Terminal and image-backed tasks: [terminal.md](references/terminal.md)
- Repository-level software engineering: [software_engineering.md](references/software_engineering.md)
- Tool calling through MCP or APIs: [tool_calling.md](references/tool_calling.md)
- Browser workflows: [browser_use.md](references/browser_use.md)
- Desktop (computer-use) workflows: [computer_use.md](references/computer_use.md)
- GPU kernels: [gpu_kernels.md](references/gpu_kernels.md)
- Professional workflows: [professional_workflows.md](references/professional_workflows.md)

Read only the assigned domain's reference and the runtime documentation supplied with the
episode. Benchmarks such as SWE-bench, Toolathlon, Online-Mind2Web, OSWorld and KernelBench
illustrate the kinds of workflows to target; do not port their tasks or depend on their services
unless the assignment asks for that integration. If the runtime lacks a capability the domain
needs, report it instead of silently switching domains. This guidance does not by itself
establish that a runtime is available.

Keep reference solutions and private checks out of the public image and setup inputs. Write a
deterministic, finite reward in [0, 1]. Check real task success, not solution markers, action
counts or the mere existence of an output file. Dependencies must already be present at solve
and verify time. Use an isolated interpreter and absolute test paths where possible, and never
import verifier code from the solver's writable working directory.

Check the task before submitting with the tools this episode provides (`aes-validate` for the
reference and empty solutions). Admission requires the reference solution to earn full reward,
a do-nothing agent to earn zero, and a reset to reproduce the initial state and a passing
reference run. Do not edit `solve.sh` to simulate the empty solution: the host runs Harbor's
`nop` agent against the same package. When the host returns feedback, read the reports and the
executed output, repair the whole task, and resubmit. Technical admission is separate from
quality review and solver calibration; do not claim either was measured when it was not.
