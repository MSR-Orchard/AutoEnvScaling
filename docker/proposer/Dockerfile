# AutoEnvScaling proposer image: the coding harnesses, Harbor, the skills and the proposer tools.
#
# Build from the repository root (the context must include autoenvscaling/ and skills/):
#
#   docker build -f docker/proposer/Dockerfile -t <registry>/autoenvscaling-proposer:<tag> .
#
# See docker/proposer/README.md for what is inside, the pinned versions, and runtime variables.
FROM python:3.12-slim

ARG NODE_VERSION=22.19.0
ARG NODE_SHA256=d36e56998220085782c0ca965f9d51b7726335aed2f5fc7321c6c0ad233aa96d
ARG PI_VERSION=0.80.6
ARG CODEX_VERSION=0.153.2
ARG CLAUDE_CODE_VERSION=latest
ARG HARBOR_VERSION=0.23.0
# Harbor's create-task skill is fetched from this commit (the revision used in the paper).
ARG HARBOR_SKILL_REPO=https://github.com/harbor-framework/harbor
ARG HARBOR_SKILL_COMMIT=f5e9d0b71ac4493a4f0620653e2913aee7fc0767

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# setsid and timeout (util-linux, coreutils) run the harness detached with a hard time budget;
# jq and ripgrep help the proposer read rollouts.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git jq procps ripgrep tar util-linux coreutils xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Node.js (checksum-verified) and the three coding harnesses.
RUN curl -fsSL -o /tmp/node.tgz \
        "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.gz" \
    && echo "${NODE_SHA256}  /tmp/node.tgz" | sha256sum -c - \
    && tar -xzf /tmp/node.tgz -C /opt \
    && mv "/opt/node-v${NODE_VERSION}-linux-x64" /opt/node \
    && rm -f /tmp/node.tgz
ENV PATH="/opt/node/bin:${PATH}"
RUN npm install -g --ignore-scripts "@earendil-works/pi-coding-agent@${PI_VERSION}" \
    && npm install -g "@openai/codex@${CODEX_VERSION}" \
    && npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
    && ln -sf /opt/node/bin/node /opt/node/bin/npm /opt/node/bin/pi /opt/node/bin/codex \
        /opt/node/bin/claude /usr/local/bin/ \
    && node --version && pi --version && codex --version && claude --version

# Harbor (with Modal, for child containers) and the autoenvscaling package, which provides the
# solver runner behind the solver_model tool.
RUN pip install "harbor[modal]==${HARBOR_VERSION}"
COPY pyproject.toml README.md /opt/aes/src/
COPY autoenvscaling /opt/aes/src/autoenvscaling
RUN pip install /opt/aes/src && harbor --version

# Skills. Harbor's create-task is fetched at the pinned commit together with Harbor's LICENSE
# (Apache-2.0) and a SOURCE file recording where it came from; it is not vendored in this repo.
RUN git init -q /tmp/harbor \
    && git -C /tmp/harbor remote add origin "${HARBOR_SKILL_REPO}" \
    && git -C /tmp/harbor fetch -q --depth 1 origin "${HARBOR_SKILL_COMMIT}" \
    && git -C /tmp/harbor checkout -q FETCH_HEAD -- skills/create-task LICENSE \
    && mkdir -p /opt/aes/skills/create-task \
    && cp -R /tmp/harbor/skills/create-task/. /opt/aes/skills/create-task/ \
    && cp /tmp/harbor/LICENSE /opt/aes/skills/create-task/LICENSE \
    && printf 'repository: %s\ncommit: %s\npath: skills/create-task\nlicense: Apache-2.0 (LICENSE copied from the same commit)\n' \
        "${HARBOR_SKILL_REPO}" "${HARBOR_SKILL_COMMIT}" > /opt/aes/skills/create-task/SOURCE \
    && test -f /opt/aes/skills/create-task/SKILL.md \
    && rm -rf /tmp/harbor
# Benchmark blocklist for git. The proposer must not read Terminal-Bench or its GitHub repositories;
# the web tools refuse those URLs (AES_BLOCKED_URL_PATTERNS) and these rules make `git clone` of
# the repositories fail. GitHub itself stays reachable, so a raw download by other means is not
# prevented here; enforce stricter rules in the sandbox infrastructure if needed.
RUN for repo in laude-institute/terminal-bench harbor-framework/terminal-bench; do \
        git config --system --add url."https://blocked.invalid/".insteadOf "https://github.com/${repo}"; \
        git config --system --add url."https://blocked.invalid/".insteadOf "git@github.com:${repo}"; \
    done
COPY skills/auto_env_scaling /opt/aes/skills/auto_env_scaling
COPY skills/harbor_domains /opt/aes/skills/harbor_domains

# Proposer tools: aes-validate, solver_model, web_search, web_fetch.
COPY docker/proposer/tools /opt/aes/tools
RUN chmod a+x /opt/aes/tools/aes-validate /opt/aes/tools/solver_model \
        /opt/aes/tools/web_search /opt/aes/tools/web_fetch \
    && chmod a+x /opt/aes/skills/auto_env_scaling/verifier_template.sh
ENV PATH="/opt/aes/tools:${PATH}"

WORKDIR /workspace
