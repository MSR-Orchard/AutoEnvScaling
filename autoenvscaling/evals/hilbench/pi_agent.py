"""Pi with the ``ask_human`` command, for the ask_human setting of HiL-style tasks.

Same agent as ``autoenvscaling.evals.pi_agent.PiAgent``; when ``ask_human_url`` is set it also
installs the ``ask_human`` command in the task container and appends the "Available Commands"
block to the instruction. The ask-human server itself runs outside the solver's container (a
sidecar), so the registry and its answers are never readable from where the agent works.
For the full_info and baseline settings, use ``PiAgent`` directly.
"""

from __future__ import annotations

from typing import Any

from harbor.environments.base import BaseEnvironment

from autoenvscaling.evals.hilbench.ask_human import INSTRUCTION_BLOCK, install_client_script
from autoenvscaling.evals.pi_agent import PiAgent

HIL_PI_AGENT_IMPORT = "autoenvscaling.evals.hilbench.pi_agent:HilPiAgent"


class HilPiAgent(PiAgent):
    def __init__(self, logs_dir, model_name: str | None = None, *,
                 ask_human_url: str | None = None, **kwargs: Any) -> None:
        if ask_human_url:
            kwargs["instruction_suffix"] = kwargs.get("instruction_suffix", "") + INSTRUCTION_BLOCK
        super().__init__(logs_dir, model_name, **kwargs)
        self._ask_human_url = ask_human_url

    @staticmethod
    def name() -> str:
        return "aes-pi-hil"

    async def setup(self, environment: BaseEnvironment) -> None:
        await super().setup(environment)
        if not self._ask_human_url:
            return
        result = await environment.exec(command=install_client_script(),
                                        user="root", timeout_sec=60)
        if result.return_code != 0:
            raise RuntimeError(f"ask_human install failed rc={result.return_code}: "
                               f"{(result.stderr or result.stdout or '')[-300:]}")
        # The command needs its URL at run time; the agent's exec env carries it.
        self._extra_env = {**self._extra_env, "ASK_HUMAN_URL": self._ask_human_url}
