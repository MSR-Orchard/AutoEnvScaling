# Proposer image

The image the proposer sandbox runs (`settings.sandbox.proposer_image`). It packages the coding
harnesses, Harbor, the skills and the proposer's tools. Everything else arrives at launch: the
workspace, the assignment, credentials and settings (Appendix "Proposer Setup" of the paper).

## Build

From the repository root:

```bash
docker build -f docker/proposer/Dockerfile -t <registry>/autoenvscaling-proposer:<tag> .
docker push <registry>/autoenvscaling-proposer:<tag>
export AES_SANDBOX_PROPOSER_IMAGE=<registry>/autoenvscaling-proposer:<tag>
```

Modal pulls the image from the registry. Pin a digest (`image@sha256:...`) for reproducible
runs. For a private registry, create a Modal Secret with `REGISTRY_USERNAME` and
`REGISTRY_PASSWORD` and pass its name to `ModalBackend(registry_secret=...)`.

## Contents

| Path | What |
|---|---|
| `/opt/node`, `/usr/local/bin/{node,npm,pi,codex,claude}` | Node.js and the harnesses |
| Python `harbor[modal]`, `autoenvscaling` | Harbor CLI for child containers; the solver runner |
| `/opt/aes/skills/create-task/` | Harbor's official skill, fetched at build time |
| `/opt/aes/skills/auto_env_scaling/` | our overlay skill, fraction-scoring verifier template and `run_agent.py` execution helper |
| `/opt/aes/skills/harbor_domains/` | guidance for non-terminal domains |
| `/opt/aes/tools/` | `aes-validate`, `solver_model`, `web_search`, `web_fetch` (see `tools/README.md`) |
| `/workspace` | the working directory; the host uploads the workspace here |

At each episode the host copies `/opt/aes/skills` into `/workspace/skills` and, on the first
round, `/opt/aes/tools` into `/workspace/tools`, which then persists across rounds as the
proposer's own editable copy.

## Pinned versions

| Component | Version | Build argument |
|---|---|---|
| Base image | `python:3.12-slim` | |
| Node.js | 22.19.0 (SHA-256 checked) | `NODE_VERSION`, `NODE_SHA256` |
| Pi coding agent (`@earendil-works/pi-coding-agent`) | 0.80.6 | `PI_VERSION` |
| Codex (`@openai/codex`) | 0.153.2 | `CODEX_VERSION` |
| Claude Code (`@anthropic-ai/claude-code`) | latest at build time | `CLAUDE_CODE_VERSION` |
| Harbor | 0.23.0 | `HARBOR_VERSION` |
| Harbor `create-task` skill | commit `f5e9d0b71ac4493a4f0620653e2913aee7fc0767` of `harbor-framework/harbor` | `HARBOR_SKILL_COMMIT` |

Pin `CLAUDE_CODE_VERSION` to a specific release for reproducible images.

## Third-party skill licence

The `create-task` skill is part of Harbor (https://github.com/harbor-framework/harbor), licensed
under Apache-2.0. It is not copied into this repository; the image build fetches
`skills/create-task` and Harbor's `LICENSE` from the pinned commit and records the source in
`/opt/aes/skills/create-task/SOURCE`. Anyone redistributing the built image redistributes that
skill under Harbor's licence.

## Runtime configuration

The host passes these with each command (see `autoenvscaling/proposer/episode.py`):

| Variable | For |
|---|---|
| `ANTHROPIC_API_KEY` (or `CLAUDE_CODE_OAUTH_TOKEN`), `ANTHROPIC_BASE_URL` | Claude Code |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `AES_CODEX_WIRE_API` | Codex |
| `AES_PROPOSER_BASE_URL`, `AES_PROPOSER_API_KEY`, `AES_PROPOSER_*` | Pi (any OpenAI-compatible endpoint) |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | child containers (`harbor run -e modal`) |
| `AES_SOLVER_BASE_URL`, `AES_SOLVER_MODEL`, `AES_SOLVER_API_KEY` | `solver_model` |
| `SERPER_API_KEY` | `web_search` |
| `AES_BLOCKED_HOSTS`, `AES_BLOCKED_URL_PATTERNS` | the web tools' blocklist |
| `AES_SANDBOX_HARBOR_ENV`, `AES_VALIDATION_*`, `AES_CALIBRATION_*` | tool settings |

Credentials can instead be kept in Modal Secrets and attached to the sandbox by name with
`AES_PROPOSER_SECRETS`.

## Network restrictions

The paper's proposer has open web access with Terminal-Bench and other benchmark pages blocked.
Modal cannot block individual hostnames, so this release blocks them best-effort:

- the sandbox maps each blocked host (`AES_PROPOSER_BLOCKED_HOSTS`, default `tbench.ai` and
  `www.tbench.ai`) to 0.0.0.0 in `/etc/hosts` at startup;
- `web_search` drops results and `web_fetch` refuses URLs on blocked hosts or matching blocked
  patterns (`AES_BLOCKED_URL_PATTERNS`, default `terminal-bench,terminal_bench,tbench`);
- Claude Code runs without its built-in WebSearch and WebFetch tools.

A process running as root in the sandbox can undo the `/etc/hosts` entries or connect by IP
address, so this is a guard against accidental access, not isolation. The host's 13-gram
contamination check is the enforced safeguard. For enforced egress control, use Modal's domain
allowlist (`ModalBackend(domain_allowlist=[...])`), at the cost of open web search.

The proposer also holds Modal credentials so that it can start child containers; anything it
starts is billed to that Modal account.
