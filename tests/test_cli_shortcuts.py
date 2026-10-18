"""`aes <step>` forwards to the selfplay group; --config may follow the step name."""
import pytest

from autoenvscaling import cli


@pytest.mark.parametrize("rest, expected", [
    (["--pool-dir", "P"], ["generate", "--pool-dir", "P"]),
    (["--config", "c.toml", "--pool-dir", "P"], ["--config", "c.toml", "generate", "--pool-dir", "P"]),
    (["--pool-dir", "P", "--config=c.toml"], ["--config=c.toml", "generate", "--pool-dir", "P"]),
])
def test_selfplay_argv_moves_config_before_step(rest, expected):
    assert cli._selfplay_argv("generate", rest) == expected


@pytest.mark.parametrize("step", cli.SELFPLAY_STEPS)
def test_steps_dispatch_to_selfplay(step, monkeypatch):
    seen = {}
    import autoenvscaling.selfplay.__main__ as selfplay
    monkeypatch.setattr(selfplay, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    assert cli.main([step, "--config", "x.toml", "--flag"]) == 0
    assert seen["argv"] == ["--config", "x.toml", step, "--flag"]


@pytest.mark.parametrize("step", cli.SELFPLAY_STEPS)
def test_step_help_is_the_selfplay_help(step, capsys):
    with pytest.raises(SystemExit) as exit_:
        cli.main([step, "--help"])
    assert exit_.value.code == 0
    assert step in capsys.readouterr().out


def test_steps_do_not_shadow_other_commands():
    assert not set(cli.SELFPLAY_STEPS) & (set(cli.COMMANDS) | {"admit"})


def test_long_form_still_works(monkeypatch):
    import autoenvscaling.selfplay.__main__ as selfplay
    seen = {}
    monkeypatch.setattr(selfplay, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    assert cli.main(["selfplay", "--config", "x.toml", "generate"]) == 0
    assert seen["argv"] == ["--config", "x.toml", "generate"]
