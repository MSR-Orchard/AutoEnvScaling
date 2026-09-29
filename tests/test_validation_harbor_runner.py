import json
import tomllib
from pathlib import Path

from autoenvscaling.validation import harbor_runner as H

T0 = "2026-09-01T00:00:00Z"


def _result(*, reward=None, exc=None, msg="", env=True, agent_setup=True, agent_exec=True,
            rewards=None):
    data = {"trial_name": "task__abc", "environment_setup": {"started_at": T0} if env else None,
            "agent_setup": {"started_at": T0} if agent_setup else None,
            "agent_execution": {"started_at": T0} if agent_exec else None}
    if rewards is not None:
        data["verifier_result"] = {"rewards": rewards}
    elif reward is not None:
        data["verifier_result"] = {"rewards": {"reward": reward}}
    if exc:
        data["exception_info"] = {"exception_type": exc, "exception_message": msg}
    return data


def test_scored_trial_reads_reward_and_ctrf_counts(tmp_path):
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    (verifier / "ctrf.json").write_text(json.dumps(
        {"results": {"summary": {"tests": 4, "passed": 3, "failed": 1}}}))
    (verifier / "test-stdout.txt").write_text("3 passed, 1 failed\n")
    out = H.parse_trial(_result(reward=0.75), tmp_path)
    assert out.scored and out.reward == 0.75
    assert (out.tests_passed, out.tests_total) == (3, 4)
    assert "3 passed" in out.verifier_output


def test_junit_counts_do_not_credit_skipped_tests(tmp_path):
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    (verifier / "junit.xml").write_text(
        '<testsuites><testsuite tests="5" failures="1" errors="0" skipped="2"/></testsuites>')
    assert H.read_test_counts(verifier) == (2, 5)


def test_single_reward_json_key_is_used_when_no_reward_key():
    out = H.parse_trial(_result(rewards={"score": 0.5}), None)
    assert out.reward == 0.5


def test_build_failure_is_a_task_failure_not_infra():
    out = H.parse_trial(_result(exc="ImageBuildError", msg="Image build for im-1 failed.",
                                agent_setup=False, agent_exec=False), None)
    assert out.build_failed and not out.infra_error and out.reward is None
    timeout = H.parse_trial(_result(exc="EnvironmentStartTimeoutError", msg="timed out",
                                    agent_setup=False, agent_exec=False), None)
    assert timeout.build_failed


def test_builder_shutdown_and_unknown_environment_errors_are_infra():
    shutdown = H.parse_trial(_result(exc="ImageBuildError", agent_setup=False, agent_exec=False,
                                     msg="Image build for im-1 terminated due to external "
                                         "shut-down."), None)
    assert shutdown.infra_error
    unknown = H.parse_trial(_result(exc="RuntimeError", msg="sandbox vanished",
                                    agent_setup=False, agent_exec=False), None)
    assert unknown.infra_error and not unknown.build_failed
    docker = H.parse_trial(_result(exc="RuntimeError", msg="docker compose build failed: exit 1",
                                   agent_setup=False, agent_exec=False), None)
    assert docker.build_failed


def test_endpoint_failures_are_infra_and_drop_the_reward():
    out = H.parse_trial(_result(reward=0.0, exc="AgentAuthenticationError", msg="401"), None)
    assert out.infra_error and out.reward is None and not out.scored
    msg = H.parse_trial(_result(reward=0.0, exc="NonZeroAgentExitCodeError",
                                msg="No API key found for provider"), None)
    assert msg.infra_error


def test_solver_timeouts_and_crashes_with_a_graded_container_still_count():
    timeout = H.parse_trial(_result(reward=0.25, exc="AgentTimeoutError", msg="1200s"), None)
    assert timeout.scored and timeout.reward == 0.25
    crash = H.parse_trial(_result(reward=0.5, exc="NonZeroAgentExitCodeError", msg="exit 1"), None)
    assert crash.scored
    ungraded = H.parse_trial(_result(exc="NonZeroAgentExitCodeError", msg="exit 1"), None)
    assert ungraded.infra_error


def test_broken_verifier_is_a_task_failure():
    out = H.parse_trial(_result(exc="RewardFileNotFoundError", msg="no reward file"), None)
    assert out.failed_phase == "verifier" and not out.infra_error and out.reward is None


def _task(root: Path, toml: str) -> Path:
    for rel in ("environment", "solution", "tests"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    (root / "task.toml").write_text(toml)
    (root / "instruction.md").write_text("Do the thing.\n")
    (root / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    return root


def test_stage_task_overrides_only_the_copy(tmp_path):
    src = _task(tmp_path / "src", '[task]\nname = "my-task"\n\n[environment]\nnetwork_mode = '
                '"no-network"\n\n[agent]\nnetwork_mode = "no-network"\n')
    staged = H.stage_task(src, tmp_path / "run" / "task", instruction="New text",
                          public_network=True)
    cfg = tomllib.loads((staged / "task.toml").read_text())
    assert cfg["task"]["name"] == "aes/my-task"
    assert cfg["environment"]["network_mode"] == "public"
    assert cfg["environment"]["build_timeout_sec"] == H.DEFAULT_BUILD_TIMEOUT_SEC
    assert "network_mode" not in cfg["agent"]
    assert (staged / "instruction.md").read_text() == "New text"
    assert "no-network" in (src / "task.toml").read_text()
    assert (src / "instruction.md").read_text() == "Do the thing.\n"


def test_stage_task_keeps_network_and_explicit_build_timeout(tmp_path):
    src = _task(tmp_path / "src", '[environment]\nnetwork_mode = "no-network"\n'
                'build_timeout_sec = 300.0\n')
    cfg = tomllib.loads((H.stage_task(src, tmp_path / "t") / "task.toml").read_text())
    assert cfg["environment"] == {"network_mode": "no-network", "build_timeout_sec": 300.0}


def test_job_config_uses_harbor_models(tmp_path):
    runner = H.HarborRunner(tmp_path, env_type="modal", max_concurrent=2, max_retries=1)
    agent = H.AgentSpec(name="pi", model_name="openai/solver-9b", kwargs={"model_api": "x"},
                        env={"OPENAI_API_KEY": "k"}, timeout_sec=90,
                        extra_allowed_hosts=("solver.example.org",))
    cfg = runner._job_config(tmp_path / "task", agent, n_attempts=4, jobs_dir=tmp_path / "jobs",
                             job_name="job")
    assert cfg.n_attempts == 4 and cfg.n_concurrent_trials == 2
    assert cfg.environment.type.value == "modal"
    a = cfg.agents[0]
    assert (a.name, a.model_name, a.override_timeout_sec) == ("pi", "openai/solver-9b", 90.0)
    assert a.extra_allowed_hosts == ["solver.example.org"]
    assert "ImageBuildError" in cfg.retry.exclude_exceptions
    assert "AgentTimeoutError" in cfg.retry.exclude_exceptions
    ref = runner._job_config(tmp_path / "task", H.AgentSpec.empty(), n_attempts=1,
                             jobs_dir=tmp_path / "jobs", job_name="job")
    assert ref.agents[0].name == "nop" and ref.agents[0].model_name is None


def test_build_log_tail_includes_error_and_trial_log(tmp_path):
    (tmp_path / "trial.log").write_text("step 3/5 RUN apt-get install nosuchpkg\nE: Unable\n")
    out = H.TrialOutcome(trial_name="t", trial_dir=tmp_path, exception_type="ImageBuildError",
                         exception_message="Image build failed", failed_phase="environment")
    text = H.build_log_tail(out)
    assert "ImageBuildError: Image build failed" in text and "nosuchpkg" in text


def test_stage_task_allowlist_merges_task_hosts(tmp_path):
    import tomllib

    from autoenvscaling.validation.harbor_runner import stage_task

    def staged_env(task_env: dict, **kwargs) -> dict:
        src = tmp_path / f"src{len(list(tmp_path.iterdir()))}"
        (src / "environment").mkdir(parents=True)
        lines = ['[task]\nname = "demo"\n', "[environment]\n"]
        for key, value in task_env.items():
            lines.append(f"{key} = {value!r}\n".replace("'", '"'))
        (src / "task.toml").write_text("".join(lines))
        dest = stage_task(src, tmp_path / (src.name + "-staged"), **kwargs)
        return tomllib.loads((dest / "task.toml").read_text())["environment"]

    # A task that already allowlists hosts keeps them, plus the solver's.
    env = staged_env({"network_mode": "allowlist", "allowed_hosts": ["pypi.org"]},
                     allowed_hosts=("solver.test", "registry.npmjs.org"))
    assert env["network_mode"] == "allowlist"
    assert env["allowed_hosts"] == ["pypi.org", "registry.npmjs.org", "solver.test"]
    # A public task is narrowed to the solver's hosts.
    env = staged_env({"network_mode": "public"}, allowed_hosts=("solver.test",))
    assert env["network_mode"] == "allowlist" and env["allowed_hosts"] == ["solver.test"]
    # Without allowed_hosts the task's own policy is untouched.
    assert staged_env({"network_mode": "public"})["network_mode"] == "public"
