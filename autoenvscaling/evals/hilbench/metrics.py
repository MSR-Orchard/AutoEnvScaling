"""HiL-Bench metrics: task success, Ask-F1, ask rate, and the shaped clarification reward.

Every number comes from two sources per episode: the task verifier's reward, and the ask-human
server's metrics file (``ask_human_metrics.json``: questions asked, blockers resolved,
precision, recall, F1). Training reward and evaluation metric therefore come from one place.

* Task success: the fraction of episodes (or, for pass@k, of tasks) whose reward reaches 1.
* Ask-F1: the mean over ask_human episodes of the server's per-episode F1. An episode in which
  the model never asked writes no metrics and counts as 0, which is exactly what the metric
  measures.
* Ask rate: the fraction of ask_human episodes with at least one question. Report it next to
  Ask-F1: a low F1 with a low ask rate means the model does not ask, a low F1 with a high ask
  rate points at the questions or at the judge.

Infrastructure failures are excluded, never scored as 0. A software engineering episode whose
verifier collected no tests is a broken verifier, not a model failure, and is excluded too.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

INFRA_STOPS = frozenset({"setup_failed", "pi_install_failed", "not_started", "error",
                         "driver_error", "infra_error"})

# Shaped reward (HiL-Bench, Appendix G.2): per question +0.3 if it targets a registered blocker,
# -0.1 if it is irrelevant or repeats a resolved blocker; plus, if at least one blocker was
# discovered, the fraction of blockers discovered. There is no task-success term.
STEP_HIT = 0.3
STEP_MISS = -0.1
CANT_ANSWER = "can't answer (perhaps transient hiccup)"


@dataclass
class HilEpisode:
    task: str
    mode: str                          # "full_info", "ask_human" or "baseline"
    reward: float
    task_type: str = "swe"             # "swe" or "sql"
    sample: int = 0
    stop_reason: str = "completed"
    ask_log: dict[str, Any] | None = None
    tests_collected: int | None = None  # None when the verifier does not report a test count
    model: str = ""
    ts: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_infra(self) -> bool:
        return self.stop_reason in INFRA_STOPS

    @property
    def verifier_ok(self) -> bool:
        return self.tests_collected is None or self.tests_collected > 0

    @classmethod
    def from_record(cls, record: dict[str, Any], model: str = "") -> HilEpisode:
        """Read one line of an ``episodes.jsonl`` written by the research harness."""
        verify = record.get("verify") or {}
        collected = verify.get("n_tests_reported") if record.get("task_type") == "swe" else None
        return cls(
            task=record["task"], mode=record["mode"], reward=float(record.get("reward") or 0.0),
            task_type=record.get("task_type", "swe"), sample=int(record.get("sample", 0)),
            stop_reason=record.get("stop_reason") or "completed",
            ask_log=verify.get("ask_log") if isinstance(verify.get("ask_log"), dict) else None,
            tests_collected=int(collected) if collected is not None else None,
            model=model or record.get("model", ""), ts=float(record.get("ts") or 0.0),
        )


@dataclass
class RewardBreakdown:
    total: float
    step_total: float
    terminal: float
    n_questions: int
    n_hits: int
    n_misses: int
    n_transient: int
    blockers_resolved: int
    n_blockers: int


def episode_reward(ask_log: dict[str, Any] | None) -> RewardBreakdown:
    """Shaped clarification reward for one episode, from the ask-human server's metrics.

    Three details matter. Hits pay three times what misses cost, so asking is worth trying
    (symmetric penalties train silence). The terminal term needs at least one discovered blocker,
    otherwise never asking would dominate asking and missing. Repeats are misses because the
    server answers at most two questions per blocker. One deliberate change from HiL-Bench: a
    "can't answer" reply means the judge failed, so it is left out of the step sum rather than
    charged -0.1.
    """
    if not isinstance(ask_log, dict):
        return RewardBreakdown(0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0)
    questions = [q for q in ask_log.get("questions") or [] if isinstance(q, dict)]
    n_blockers = int(ask_log.get("n_blockers") or 0)
    hits = misses = transient = 0
    for q in questions:
        if q.get("response") == CANT_ANSWER:
            transient += 1
        elif q.get("blocker_name"):
            hits += 1
        else:
            misses += 1
    step_total = STEP_HIT * hits + STEP_MISS * misses
    # The server keeps resolved blockers as a set, so two hits on one blocker count once for
    # recall while both earn the step reward.
    resolved_list = ask_log.get("blockers_resolved_list")
    resolved = (len(resolved_list) if isinstance(resolved_list, list)
                else int(ask_log.get("blockers_resolved") or 0))
    terminal = resolved / n_blockers if resolved >= 1 and n_blockers > 0 else 0.0
    return RewardBreakdown(step_total + terminal, step_total, terminal, len(questions), hits,
                           misses, transient, resolved, n_blockers)


def is_trainable(episode: HilEpisode) -> tuple[bool, str]:
    """Whether an episode may enter an RL batch for the clarification reward.

    In training, an infrastructure failure read as reward 0 teaches the policy that what it did
    was worthless, so the rule is stricter than for evaluation.
    """
    if episode.is_infra:
        return False, f"infrastructure failure: {episode.stop_reason}"
    if episode.mode != "ask_human":
        return False, "not an ask_human episode"
    if not isinstance(episode.ask_log, dict):
        return False, "no ask_human metrics: the ask-human server produced none"
    if int(episode.ask_log.get("n_blockers") or 0) <= 0:
        return False, "ask_human metrics report 0 blockers: the registry never loaded"
    return True, ""


def dedupe(episodes: Iterable[HilEpisode]) -> list[HilEpisode]:
    """One episode per (model, task type, mode, task, sample), preferring the most usable one.

    The model is part of the key: without it, merging runs of two models lets the higher reward
    of either win and silently mixes them into one table.
    """
    def rank(e: HilEpisode) -> tuple:
        return (not e.is_infra, e.verifier_ok, e.reward, e.ts)

    chosen: dict[tuple, HilEpisode] = {}
    for e in episodes:
        key = (e.model, e.task_type, e.mode, e.task, e.sample)
        if key not in chosen or rank(e) > rank(chosen[key]):
            chosen[key] = e
    return list(chosen.values())


def summarize(episodes: Iterable[HilEpisode], mode: str, *, task_type: str | None = None,
              exclude_stop_reasons: Iterable[str] = (), success_threshold: float = 1.0
              ) -> dict[str, Any] | None:
    """Metrics for one mode (and optionally one task type) over a set of episodes.

    ``exclude_stop_reasons`` drops further stop reasons besides infrastructure failures (for
    example agent timeouts, if a table was computed that way); the count is reported.
    """
    extra_excluded = set(exclude_stop_reasons)
    selected = [e for e in episodes
                if e.mode == mode and (task_type is None or e.task_type == task_type)]
    if not selected:
        return None
    usable = [e for e in selected
              if not e.is_infra and e.verifier_ok and e.stop_reason not in extra_excluded]
    by_task: dict[str, list[HilEpisode]] = defaultdict(list)
    for e in usable:
        by_task[e.task].append(e)
    out: dict[str, Any] = {
        "mode": mode,
        "task_type": task_type or "all",
        "n_episodes": len(selected),
        "n_episodes_scored": len(usable),
        "n_infra": sum(e.is_infra for e in selected),
        "n_verifier_broken": sum(not e.is_infra and not e.verifier_ok for e in selected),
        "n_excluded_other": sum(e.stop_reason in extra_excluded for e in selected
                                if not e.is_infra and e.verifier_ok),
        "n_tasks": len({e.task for e in selected}),
        "n_tasks_scored": len(by_task),
        "success_rate": (statistics.fmean(e.reward >= success_threshold for e in usable)
                         if usable else None),
        "pass_at_k": (statistics.fmean(any(e.reward >= success_threshold for e in es)
                                       for es in by_task.values()) if by_task else None),
    }
    if mode == "ask_human":
        # Ask metrics come from the ask-human server, not the verifier, so an episode whose
        # verifier collected nothing still counts here.
        pool = [e for e in selected if not e.is_infra and e.stop_reason not in extra_excluded]
        f1s: list[float] = []
        asked = questions = resolved = opportunities = hits = 0
        for e in pool:
            log = e.ask_log
            if isinstance(log, dict) and "f1" in log:
                f1s.append(float(log["f1"]))
                n_q = int(log.get("total_questions") or 0)
                questions += n_q
                asked += int(n_q > 0)
                resolved += int(log.get("blockers_resolved") or 0)
                opportunities += int(log.get("n_blockers") or 0)
                hits += sum(1 for q in log.get("questions") or []
                            if isinstance(q, dict) and q.get("blocker_name"))
            else:
                f1s.append(0.0)  # never asked: a real zero
        n = len(f1s)
        out.update({
            "ask_f1": statistics.fmean(f1s) if n else None,
            "ask_rate": asked / n if n else None,
            "episodes_that_asked": asked,
            "questions_asked": questions,
            "question_precision": hits / questions if questions else None,
            "blockers_resolved": resolved,
            "blocker_recall": resolved / opportunities if opportunities else None,
            "ask_n_episodes": n,
            "mean_shaped_reward": (statistics.fmean(episode_reward(e.ask_log).total
                                                    for e in pool) if pool else None),
        })
    return out


def report(episodes: Iterable[HilEpisode], **kwargs: Any) -> dict[str, Any]:
    """Per model: every (task type, mode) cell that has episodes."""
    episodes = dedupe(episodes)
    by_model: dict[str, list[HilEpisode]] = defaultdict(list)
    for e in episodes:
        by_model[e.model or "unknown-model"].append(e)
    result: dict[str, Any] = {}
    for model, eps in sorted(by_model.items()):
        cells = {}
        for task_type in (None, "sql", "swe"):
            for mode in ("full_info", "ask_human", "baseline"):
                cell = summarize(eps, mode, task_type=task_type, **kwargs)
                if cell:
                    cells[f"{task_type or 'all'}.{mode}"] = cell
        result[model] = cells
    return result
