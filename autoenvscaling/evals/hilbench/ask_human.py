"""The ask_human tool: HiL-Bench's ask-human server, its judge, and the Pi-side command.

Server. Questions are answered by HiL-Bench's own ask-human MCP server, unchanged. It loads the
blocker registry, asks a judge model whether a question targets one blocker, returns that
blocker's resolution (at most twice per blocker) or "irrelevant question", and writes
``ask_human_metrics.json`` (questions, blockers resolved, precision, recall, F1). Ask-F1 and the
shaped training reward are both computed from that file. The server is not redistributed here:
``upstream_server`` takes it from a HiL-Bench checkout and checks it against the pinned hash,
because editing it would change the benchmark's tool surface.

Judge. The paper's runs use DeepSeek-V4-Flash (``deepseek-v4-flash-0731``, through OpenRouter)
as the judge that matches questions to blockers, not HiL-Bench's Llama-3.3-70B. Ask-F1 numbers
from this setup are therefore not comparable with the HiL-Bench leaderboard. Task success does
not depend on the judge.

Client. The Pi agent has no MCP client, so the agent gets a shell command, ``ask_human
"<question>"`` (``ask_human_client.py``), and one "Available Commands" block appended to the
instruction, mirroring how HiL-Bench surfaces the tool for its text-to-SQL tasks. For software
engineering tasks this replaces a first-class MCP tool and is a recorded deviation.
"""

from __future__ import annotations

import hashlib
from importlib import resources
from pathlib import Path

HIL_BENCH_REPO = "https://github.com/hilbenchauthors/hil-bench"
HIL_BENCH_COMMIT = "352d14c861f2531949dfa91848d4b2fe46b8a247"
UPSTREAM_SERVER_RELPATH = Path("harbor_swe/shared/mcp-servers/ask-human/server.py")
UPSTREAM_SERVER_SHA256 = "f1af5097ba8aebfd7fdb4aae802a61585b68a74b726f644ff16f5d92ae0c16a9"
# Python packages the upstream server imports, at the versions the paper's runs used.
UPSTREAM_SERVER_REQUIREMENTS = ("fastmcp==2.12.2", "litellm>=1.75.5,<1.76.0", "pydantic==2.11.7")

# The judge used for every Ask-F1 number in the paper (an OpenRouter model id), and a description
# of the official HiL-Bench judge it replaced.
PAPER_JUDGE_MODEL = "deepseek/deepseek-v4-flash-0731"
OFFICIAL_JUDGE = "Llama-3.3-70B (HiL-Bench's official judge)"

# Sentinel replies of the upstream server.
CANT_ANSWER = "can't answer (perhaps transient hiccup)"
IRRELEVANT_QUESTION = "irrelevant question"
MAX_ANSWERED_QUESTIONS_PER_BLOCKER = 2
METRICS_FILENAME = "ask_human_metrics.json"

DEFAULT_URL = "http://ask-human:8000/mcp"
CLIENT_INSTALL_PATH = "/usr/local/bin/ask_human"

# Backticks and $(...) in a shell argument are expanded by the caller's shell before the command
# sees them, which silently deletes the most specific words of a question. The software
# engineering block always carries this warning.
SHELL_QUOTING_WARNING = (
    "  Write questions as plain prose: backticks and $(...) are interpreted by "
    "the shell and their contents will be stripped from your question.\n"
)
INSTRUCTION_BLOCK = (
    "\n\n# Available Commands\n"
    "The following command is installed in this terminal - run it directly from the shell.\n"
    "- `ask_human \"<question>\"` - ask a specific clarifying question about a task blocker\n"
    + SHELL_QUOTING_WARNING
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upstream_server(hil_bench_root: Path, *, expected_sha256: str = UPSTREAM_SERVER_SHA256) -> Path:
    """Path of HiL-Bench's ask-human server in a checkout, after checking its hash.

    HiL-Bench also has a different program named ``ask_human_server.py`` (a Flask CLI with a
    different configuration contract); configuring one with the other's variables gives a judge
    that answers nothing and an Ask-F1 of exactly zero, which looks like a model that never asks.
    The hash check rules that out.
    """
    path = Path(hil_bench_root) / UPSTREAM_SERVER_RELPATH
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; check out {HIL_BENCH_REPO} at {HIL_BENCH_COMMIT}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"{path} has sha256 {actual}, expected {expected_sha256} "
                           f"(HiL-Bench commit {HIL_BENCH_COMMIT})")
    return path


def judge_base_url(url: str) -> str:
    """The upstream server appends /v1/chat/completions itself, so the URL must stop before /v1."""
    url = url.strip().rstrip("/")
    return url.removesuffix("/v1")


def server_env(*, judge_url: str, judge_model: str = PAPER_JUDGE_MODEL, judge_api_key: str = "",
               registry_path: str = "/ask-human-data/blocker_registry.json",
               output_dir: str = "/harbor_shared") -> dict[str, str]:
    """Environment for the upstream server with an OpenAI-compatible judge endpoint."""
    return {
        "ASK_HUMAN_BACKEND": "vllm",
        "VLLM_BASE_URL": judge_base_url(judge_url),
        "VLLM_API_KEY": judge_api_key,
        "VLLM_MODEL": judge_model,
        "ASK_HUMAN_MODEL": judge_model,
        "BLOCKER_REGISTRY_PATH": registry_path,
        "OUTPUT_DIR": output_dir,
    }


def client_source() -> str:
    """Source of the ``ask_human`` command, to be written into a task container."""
    return resources.files(__package__).joinpath("ask_human_client.py").read_text()


def install_client_script(path: str = CLIENT_INSTALL_PATH) -> str:
    """Shell commands that install the ``ask_human`` command in a container.

    The command reads the server URL from ``ASK_HUMAN_URL`` (default ``DEFAULT_URL``); the
    agent passes it in the environment it launches Pi with.
    """
    source = client_source()
    if "AESASKHUMAN" in source:
        raise ValueError("client source contains the heredoc delimiter")
    return (
        f"mkdir -p {Path(path).parent}\n"
        f"cat > {path} <<'AESASKHUMAN'\n{source}\nAESASKHUMAN\n"
        f"chmod 755 {path}\n"
    )
