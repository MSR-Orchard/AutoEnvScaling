"""Training with Miles (github.com/radixark/miles, patched as in third_party/miles).

    config.py     MilesConfig and presets; the paper's settings are the defaults
    launcher.py   command, environment and checkpoint discovery for a Miles job
    launch/       train.sh (inside the Miles container) and apptainer.sh (host wrapper)
    bridge.py     file handshake between the self-play loop and a running Miles job
    rollout.py    the Miles rollout function for bridge mode
    hooks.py      the reward post-process: group-mean-centred advantages

Carry mode (partial rollout and over-sampling, the paper's solver-training loop) lives in
`autoenvscaling.training.carry`. `python -m autoenvscaling.training.miles --help` prints or
runs a configured job.
"""

from autoenvscaling.training.miles.config import (
    BRIDGE,
    CARRY,
    GENERATE_FN,
    PRESETS,
    SFT,
    MilesConfig,
)
from autoenvscaling.training.miles.launcher import MilesLauncher, TrainingResult, latest_checkpoint

__all__ = ["BRIDGE", "CARRY", "GENERATE_FN", "PRESETS", "SFT", "MilesConfig", "MilesLauncher",
           "TrainingResult", "latest_checkpoint"]
