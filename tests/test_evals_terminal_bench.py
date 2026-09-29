import json
from dataclasses import dataclass

import pytest
from harbor.agents.installed.base import ApiRateLimitError
from harbor.models.agent.context import AgentContext

from autoenvscaling.evals.cli import main as cli_main
from autoenvscaling.evals.endpoint import EndpointSpec, normalize_base_url, preflight
from autoenvscaling.evals.pi_agent import (
    PiAgent,
    parse_marker,
    request_extension,
    validate_sampling,
)
from autoenvscaling.evals.terminal_bench import (
    BENCHMARKS,
    EvalSettings,
    build_job_config,
    dataset_config,
    report,
)


def test_dataset_specs_parse_like_harbor_run(tmp_path):
    reg = dataset_config("terminal-bench@2.1")
    assert reg.is_registry() and reg.name == "terminal-bench" and reg.version == "2.1"
    pkg = dataset_config("terminal-bench/terminal-bench@4.0.0")
    assert pkg.is_package() and pkg.ref == "4.0.0"
    local = dataset_config(str(tmp_path), task_names=["a*"])
    assert local.is_local() and local.task_names == ["a*"]


def test_job_config_matches_paper_eval_protocol():
    settings = EvalSettings(benchmark="tb2.1", endpoint=EndpointSpec(model="org/model-x"),
                            job_name="unit")
    config = build_job_config(settings)
    assert config.n_attempts == 5                      # avg@5
    assert config.environment.type.value == "modal"
    assert config.environment.override_cpus == 4      # Appendix G.3: 4 CPUs, 8 GiB
    assert config.environment.override_memory_mb == 8192
    agent = config.agents[0]
    assert agent.import_path == "autoenvscaling.evals.pi_agent:PiAgent"
    assert agent.model_name == "org/model-x"
    assert agent.kwargs["tools"] == "bash"
    # The key is passed by reference to a host variable, never by value.
    assert agent.env == {"AES_EVAL_API_KEY": "${AES_EVAL_API_KEY}"}
    assert "AgentTimeoutError" in config.retry.exclude_exceptions
    assert config.datasets[0].name == "terminal-bench/terminal-bench-2-1"   # Harbor package registry


def test_job_config_requires_model_and_valid_k():
    with pytest.raises(ValueError):
        build_job_config(EvalSettings(endpoint=EndpointSpec(model="")))
    with pytest.raises(ValueError):
        build_job_config(EvalSettings(endpoint=EndpointSpec(model="m"), k=0))
    assert set(BENCHMARKS) == {"tb2.1", "tb4"}


def test_agent_timeout_leaves_margin_for_harbor():
    settings = EvalSettings(endpoint=EndpointSpec(model="m"), agent_timeout_sec=1800)
    agent = build_job_config(settings).agents[0]
    assert agent.kwargs["agent_timeout_sec"] == 1800
    assert agent.override_timeout_sec > 1800


def test_cli_config_prints_job_without_running(capsys):
    assert cli_main(["config", "--benchmark", "tb4", "--model", "m", "-k", "3",
                     "--job-name", "x", "--sampling", "qwen-solving"]) == 0
    config = json.loads(capsys.readouterr().out)
    assert config["n_attempts"] == 3
    assert config["datasets"][0]["name"] == "terminal-bench/terminal-bench"
    assert config["agents"][0]["kwargs"]["sampling"]["top_k"] == 20


def test_sampling_validation():
    assert validate_sampling(None) is None
    assert validate_sampling({"temperature": 0.6, "top_k": 20.0}) == {"temperature": 0.6, "top_k": 20}
    with pytest.raises(ValueError):
        validate_sampling({"top_p": 0})
    with pytest.raises(ValueError):
        validate_sampling({"nucleus": 0.9})


def test_request_extension_pins_model_and_logs_policy():
    js = request_extension("org/model", {"temperature": 0.6}, {"order": ["P"]}, "/logs/agent/p.jsonl")
    assert '"model":"org/model"' in js and '"temperature":0.6' in js
    assert "model identity drift" in js and "/logs/agent/p.jsonl" in js


def test_markers_and_base_url():
    assert parse_marker("x\nPI_EXIT=3\nPI_EXIT=0\n", "PI_EXIT") == 0
    assert parse_marker("PI_EXIT=abc", "PI_EXIT") is None
    assert normalize_base_url("http://host:8000/") == "http://host:8000/v1"
    assert normalize_base_url("http://host/v1/") == "http://host/v1"


@dataclass
class _Result:
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""


class FakeEnvironment:
    """Answers the Pi agent's shell commands by what they contain."""

    def __init__(self, pi_exit=0, tail=""):
        self.pi_exit = pi_exit
        self.tail = tail
        self.commands = []

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.commands.append((command, env))
        if "EXPECTED_PI" in command or "models.json" in command:
            return _Result()
        if "mkfifo" in command:
            return _Result(stdout="PI_PID=4242\n")
        if "PI_RUNNING" in command:
            return _Result(stdout=f"PI_EXIT={self.pi_exit}\n")
        if "node -e" in command:
            return _Result(stdout="PI_INPUT=100\nPI_OUTPUT=20\nPI_CACHE=5\nPI_TURNS=3\n")
        if "tail -c" in command:
            return _Result(stdout=self.tail)
        return _Result()


def _agent(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setenv("AES_EVAL_BASE_URL", "http://solver.example:8000")
    return PiAgent(logs_dir=tmp_path, model_name="org/model",
                   extra_env={"AES_EVAL_API_KEY": "secret"}, **kwargs)


async def test_pi_agent_setup_and_clean_run(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch, sampling={"temperature": 0.6})
    env = FakeEnvironment()
    await agent.setup(env)
    config_cmd = env.commands[1][0]
    assert '"baseUrl":"http://solver.example:8000/v1"' in config_cmd
    assert '"apiKey":"$AES_EVAL_API_KEY"' in config_cmd
    assert '"compaction": {"enabled": false}' in config_cmd
    assert "secret" not in config_cmd
    context = AgentContext()
    await agent.run("fix the build", env, context)
    launch_cmd, launch_env = env.commands[2]
    assert "--tools bash" in launch_cmd and "--no-skills" in launch_cmd
    assert launch_env["AES_EVAL_API_KEY"] == "secret"
    assert context.n_input_tokens == 105 and context.n_output_tokens == 20
    assert context.metadata["pi_exit_code"] == 0 and context.metadata["pi_turns"] == 3


async def test_pi_agent_context_limit_is_a_model_outcome(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch)
    env = FakeEnvironment(pi_exit=1, tail="Error: maximum context length is 262144 tokens")
    context = AgentContext()
    await agent.run("task", env, context)
    assert context.metadata["context_limited"] is True


async def test_pi_agent_provider_failure_is_retryable(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch)
    env = FakeEnvironment(pi_exit=1, tail="429 Too Many Requests: rate limit")
    with pytest.raises(ApiRateLimitError):
        await agent.run("task", env, AgentContext())


async def test_pi_agent_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("AES_EVAL_BASE_URL", "http://x")
    agent = PiAgent(logs_dir=tmp_path, model_name="m")
    with pytest.raises(RuntimeError, match="AES_EVAL_API_KEY"):
        await agent.run("task", FakeEnvironment(), AgentContext())


def test_report_merges_job_dirs(tmp_path):
    for job, rewards in (("run", [1.0, None]), ("rerun", [0.0])):
        for i, r in enumerate(rewards):
            d = tmp_path / job / f"t{i}"
            d.mkdir(parents=True)
            d.joinpath("result.json").write_text(json.dumps({
                "task_name": "task-a", "trial_name": f"{job}{i}", "started_at": f"{job}{i}",
                "verifier_result": None if r is None else {"rewards": {"reward": r}},
                "exception_info": None if r is not None else {"exception_type": "RuntimeError"},
            }))
    rep = report([tmp_path / "run", tmp_path / "rerun"], k=2)
    assert rep["avg_at_k"] == pytest.approx(0.5)
    assert rep["n_infra_errors"] == 1 and rep["metric"] == "avg@2"


def test_preflight_checks_model_identity():
    import httpx

    def handler(request):
        body = json.loads(request.content)
        assert request.headers["Authorization"] == "Bearer k"
        return httpx.Response(200, json={"model": body["model"] + "-20260101",
                                         "choices": [{"message": {"content": "PREFLIGHT_OK"}}],
                                         "usage": {"prompt_tokens": 5, "completion_tokens": 2}})

    spec = EndpointSpec(model="org/m")
    env = {"AES_EVAL_BASE_URL": "http://e", "AES_EVAL_API_KEY": "k"}
    result = preflight(spec, env=env, transport=httpx.MockTransport(handler))
    assert result.returned_model == "org/m-20260101"

    def wrong(request):
        return httpx.Response(200, json={"model": "other", "choices": [{}]})

    with pytest.raises(RuntimeError, match="expected"):
        preflight(spec, env=env, transport=httpx.MockTransport(wrong))
