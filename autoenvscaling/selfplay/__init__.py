"""Training the solver, and the proposer when it shares the solver's weights (Section 3.3).

`schedule.Schedule` says when reviews, generation rounds and proposer updates happen,
`loop.SelfPlayLoop` runs them, and `trainer.Trainer` is the small interface to whatever updates
the weights (Miles in `autoenvscaling.training.miles.bridge`).
"""

from autoenvscaling.selfplay.loop import FRONTIER, SHARED, STATIC, LoopConfig, SelfPlayLoop
from autoenvscaling.selfplay.schedule import Schedule
from autoenvscaling.selfplay.trainer import FixedEndpointTrainer, StepHandle, Trainer

__all__ = ["FRONTIER", "SHARED", "STATIC", "FixedEndpointTrainer", "LoopConfig", "Schedule",
           "SelfPlayLoop", "StepHandle", "Trainer"]
