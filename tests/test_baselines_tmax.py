import json
import tomllib
import zipfile

from harbor.models.task.config import TaskConfig

from autoenvscaling.baselines import tmax as T

CONTAINER_DEF = """Bootstrap: docker
From: python:3.12-slim

%files
    /cluster/home/someone/tasks/t001/fixtures/data.csv /app/data.csv

%post
    apt-get update
    pip install pandas==2.2.2

%environment
    export APP_HOME=/app

%labels
    author tmax
"""


def _zip(path, tid="t001", with_fixture=True):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{tid}/container.def", CONTAINER_DEF)
        zf.writestr(f"{tid}/task.json", json.dumps({"description": "Clean /app/data.csv.",
                                                     "domain": "data", "skill_type": "pandas"}))
        zf.writestr(f"{tid}/test_final_state.py", "def test_clean():\n    assert True\n")
        zf.writestr(f"{tid}/test_initial_state.py", "def test_present():\n    assert True\n")
        if with_fixture:
            zf.writestr(f"{tid}/fixtures/data.csv", "a,b\n1,2\n")


def test_parse_container_def():
    image, files, post, env, unhandled = T.parse_container_def(CONTAINER_DEF)
    assert image == "python:3.12-slim"
    assert files == [("/cluster/home/someone/tasks/t001/fixtures/data.csv", "/app/data.csv")]
    assert any("pandas==2.2.2" in line for line in post)
    assert env == ["APP_HOME=/app"] and unhandled == {"labels"}


def test_convert_zip_builds_a_valid_harbor_task(tmp_path):
    _zip(tmp_path / "tasks.zip")
    results = T.convert_zip(tmp_path / "tasks.zip", tmp_path / "pool",
                            [{"task_id": "t001", "tier": "medium"}])
    assert results[0]["ok"] and results[0]["unhandled_sections"] == ["labels"]
    task = tmp_path / "pool" / "t001"
    dockerfile = (task / "environment" / "Dockerfile").read_text()
    assert dockerfile.startswith("FROM python:3.12-slim\nENV APP_HOME=/app\n")
    assert "COPY fixtures/data.csv /app/data.csv" in dockerfile
    assert "TMAX_INITIAL_STATE_WARN" in dockerfile
    assert (task / "environment" / "fixtures" / "data.csv").read_text() == "a,b\n1,2\n"
    assert "pytest-json-ctrf" in (task / "environment" / "_tmax_build.sh").read_text()
    assert (task / "instruction.md").read_text() == "Clean /app/data.csv."
    assert (task / "tests" / "test.sh").read_text() == T.MANAGED_VERIFIER.read_text()
    assert (task / "tests" / "test_outputs.py").is_file()
    config = (task / "task.toml").read_text()
    TaskConfig.model_validate_toml(config)
    parsed = tomllib.loads(config)
    assert parsed["task"]["name"] == "tmax/t001"
    assert parsed["metadata"]["tags"] == ["tmax15k", "data", "pandas", "medium"]
    assert parsed["environment"]["network_mode"] == "no-network"
    report = json.loads((tmp_path / "pool" / "_conversion_report.json").read_text())
    assert report[0]["task_id"] == "t001"


def test_missing_fixture_is_reported_and_legacy_template(tmp_path):
    _zip(tmp_path / "tasks.zip", with_fixture=False)
    results = T.convert_zip(tmp_path / "tasks.zip", tmp_path / "pool")
    assert not results[0]["ok"] and "fixture" in results[0]["error"]
    slim = T.convert_zip(tmp_path / "tasks.zip", tmp_path / "slim", slim=True, template="legacy")
    assert slim[0]["ok"]
    assert "junitxml" in (tmp_path / "slim" / "t001" / "tests" / "test.sh").read_text()
