"""Contamination check: flag an instruction that shares an n-gram with a held-out task.

Following Tmax (Ivison et al., 2026, Section 3.2 and Appendix B.4), a generated task is flagged if
any window of n = 13 consecutive tokens (sliding window, stride 1) in its instruction matches a
13-gram from the instruction of any held-out benchmark task. The check measures textual overlap
only; it does not catch paraphrased copies.

Normalisation. Tmax does not specify its tokenizer, so we use the one from our earlier analysis
(static_metrics.ngrams): lowercase the text and keep word tokens, ``re.findall(r"\\w+", text)``,
so punctuation, Markdown syntax and whitespace never affect a match. Before tokenising we drop
every line that mentions a benchmark canary string (``canary``), because Terminal-Bench style
tasks may carry an identical canary GUID that says nothing about the task itself.

Held-out sources. ``ValidationSettings.heldout_dataset`` names one or more sources separated by
commas. Each source is either a local directory (every ``instruction.md`` below it is one held-out
task) or a Harbor dataset reference such as ``terminal-bench/terminal-bench-2-1``, which is fetched
once with ``harbor download <ref> --export`` into ``<work_dir>/contamination/datasets``. The
default setting ``terminal-bench@2.1`` is mapped to that registry name (``DATASET_ALIASES``).
The n-gram index is cached as JSON in ``<work_dir>/contamination``; n-grams are stored as 64-bit
BLAKE2b digests of the space-joined tokens.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from autoenvscaling.validation.harbor_runner import harbor_cli

NORMALIZATION = "lowercase-word-tokens-v1"
# Terminal-Bench 2.1 is published in Harbor's package registry as terminal-bench/terminal-bench-2-1;
# the legacy registry that the ``name@version`` form resolves against only lists version 2.0.
DATASET_ALIASES = {"terminal-bench@2.1": "terminal-bench/terminal-bench-2-1"}
_TOKEN = re.compile(r"\w+")
_CANARY_LINE = re.compile(r"canary", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, ignoring lines that carry a benchmark canary string."""
    kept = [line for line in text.splitlines() if not _CANARY_LINE.search(line)]
    return _TOKEN.findall("\n".join(kept).lower())


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Every window of n consecutive tokens (stride 1). Empty if the text is shorter than n."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _digest(gram: tuple[str, ...]) -> str:
    return hashlib.blake2b(" ".join(gram).encode(), digest_size=8).hexdigest()


@dataclass
class Match:
    heldout_task: str
    ngram: str


@dataclass
class NgramIndex:
    """Digest of every held-out n-gram, mapped to the held-out tasks that contain it."""

    n: int
    grams: dict[str, list[str]] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    n_tasks: int = 0

    @classmethod
    def build(cls, instructions: Mapping[str, str], n: int,
              sources: Iterable[str] = ()) -> "NgramIndex":
        grams: dict[str, list[str]] = {}
        for task_name in sorted(instructions):
            for gram in {_digest(g) for g in ngrams(tokenize(instructions[task_name]), n)}:
                grams.setdefault(gram, []).append(task_name)
        return cls(n=n, grams=grams, sources=list(sources), n_tasks=len(instructions))

    def matches(self, text: str, limit: int = 20) -> list[Match]:
        """Held-out n-grams found in ``text``, at most ``limit`` distinct n-grams."""
        found: list[Match] = []
        seen: set[str] = set()
        for gram in ngrams(tokenize(text), self.n):
            key = _digest(gram)
            if key in seen or key not in self.grams:
                continue
            seen.add(key)
            for task_name in self.grams[key]:
                found.append(Match(heldout_task=task_name, ngram=" ".join(gram)))
            if len(seen) >= limit:
                break
        return found

    def to_json(self) -> str:
        return json.dumps({"n": self.n, "normalization": NORMALIZATION, "sources": self.sources,
                           "n_tasks": self.n_tasks, "grams": self.grams})

    @classmethod
    def from_json(cls, text: str) -> "NgramIndex":
        data = json.loads(text)
        if data.get("normalization") != NORMALIZATION:
            raise ValueError("cached index uses a different normalisation; rebuild it")
        return cls(n=int(data["n"]), grams=data["grams"], sources=list(data["sources"]),
                   n_tasks=int(data["n_tasks"]))


def split_sources(heldout: str) -> list[str]:
    return [s.strip() for s in heldout.split(",") if s.strip()]


def read_instructions(root: Path) -> dict[str, str]:
    """Every instruction.md under ``root``, keyed by its task directory relative to ``root``."""
    root = Path(root)
    out: dict[str, str] = {}
    for path in sorted(root.rglob("instruction.md")):
        rel = path.parent.relative_to(root).as_posix()
        out[rel if rel != "." else root.name] = path.read_text(errors="replace")
    return out


def fetch_harbor_dataset(ref: str, dest: Path, harbor_bin: str | None = None) -> Path:
    """Download a Harbor dataset once with ``harbor download <ref> --export -o <dest>``."""
    dest = Path(dest)
    if dest.is_dir() and any(dest.rglob("instruction.md")):
        return dest
    partial = dest.with_name(dest.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    cmd = [harbor_bin or harbor_cli(), "download", ref, "--export", "-o", str(partial)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"`{' '.join(cmd)}` failed ({proc.returncode}): "
                           f"{(proc.stderr or proc.stdout)[-2000:]}")
    if not any(partial.rglob("instruction.md")):
        raise RuntimeError(f"harbor download {ref} produced no instruction.md files")
    if dest.exists():
        shutil.rmtree(dest)
    partial.rename(dest)
    return dest


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.@-]+", "_", text).strip("_")[:80] or "heldout"


def load_index(heldout: str, cache_dir: Path, n: int = 13, harbor_bin: str | None = None,
               refresh: bool = False) -> NgramIndex:
    """Build, or load from cache, the n-gram index for the held-out sources in ``heldout``."""
    sources = split_sources(heldout)
    if not sources:
        raise ValueError("no held-out source configured for the contamination check")
    cache_dir = Path(cache_dir)
    key = hashlib.sha256(json.dumps([sources, n, NORMALIZATION]).encode()).hexdigest()[:12]
    cache = cache_dir / f"index-{_slug(sources[0])}-n{n}-{key}.json"
    if cache.is_file() and not refresh:
        return NgramIndex.from_json(cache.read_text())
    instructions: dict[str, str] = {}
    for source in sources:
        local = Path(source).expanduser()
        root = local if local.is_dir() else fetch_harbor_dataset(
            DATASET_ALIASES.get(source, source), cache_dir / "datasets" / _slug(source),
            harbor_bin=harbor_bin)
        for name, text in read_instructions(root).items():
            instructions[f"{source}:{name}"] = text
    if not instructions:
        raise RuntimeError(f"no held-out instructions found in {sources}")
    index = NgramIndex.build(instructions, n, sources=sources)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(index.to_json())
    tmp.replace(cache)
    return index
