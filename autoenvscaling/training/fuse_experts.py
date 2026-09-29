"""Rewrite a fine-tuned Qwen3.5/Qwen3.6 MoE checkpoint in its base checkpoint's tensor layout.

Some trainers export MoE experts one by one (``...mlp.experts.<i>.gate_proj.weight``,
``up_proj``, ``down_proj``) and drop weights they never trained (the vision tower). Miles'
checkpoint converter expects the base checkpoint's layout: stacked
``...mlp.experts.gate_up_proj`` of shape [experts, 2 * ffn, hidden] (gate rows before up rows)
and ``...mlp.experts.down_proj`` of shape [experts, hidden, ffn], with every base tensor present.

The output is driven by the base index, so it has exactly the base tensor names. Each tensor
comes from the fine-tuned checkpoint when it exists there (fusing experts in numeric order),
otherwise from the base checkpoint when its name starts with one of `base_only_prefixes`, and
anything else is an error. Every shape and dtype is checked against the base shard headers
before a shard is written; float32 tensors whose base dtype is bfloat16 are cast down (some
trainers keep linear-attention parameters in float32). Tokenizer and generation files come from
the fine-tuned checkpoint, `config.json` from the base. The result is written to `<out>.tmp`
and renamed only when complete.

    python -m autoenvscaling.training.fuse_experts --base <base hf dir> --sft <sft hf dir> --out <dir>

Needs `torch` and `safetensors` (not dependencies of this package).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

EXPERT_RE = re.compile(r"^(.*\.mlp\.experts)\.(gate_up_proj|down_proj)$")
INDEX_FILE = "model.safetensors.index.json"
SHARD_BYTES = 10 * 2**30


class LayoutError(ValueError):
    """The fine-tuned checkpoint cannot be mapped onto the base layout."""


@dataclass(frozen=True)
class Source:
    kind: str                  # "sft", "base", "stack_gate_up" or "stack_down"
    names: tuple[str, ...]     # source tensor names, in stacking order


def _expert_count(prefix: str, sft_names: set[str]) -> int:
    pattern = re.compile(re.escape(prefix) + r"\.(\d+)\.gate_proj\.weight$")
    ids = sorted(int(m.group(1)) for n in sft_names if (m := pattern.match(n)))
    if not ids or ids != list(range(len(ids))):
        raise LayoutError(f"expert ids under {prefix} are missing or not contiguous")
    return len(ids)


def plan_layout(base_names: Iterable[str], sft_names: Iterable[str], *,
                base_only_prefixes: tuple[str, ...] = ("model.visual.",)) -> tuple[dict[str, Source], set[str]]:
    """Map every base tensor to its source. Returns (plan, fine-tuned tensors left unused)."""
    sft = set(sft_names)
    plan: dict[str, Source] = {}
    used: set[str] = set()
    for name in sorted(base_names):
        m = EXPERT_RE.match(name)
        if m and name not in sft:
            prefix, kind = m.groups()
            n = _expert_count(prefix, sft)
            if kind == "gate_up_proj":
                names = tuple(x for e in range(n) for x in (f"{prefix}.{e}.gate_proj.weight",
                                                             f"{prefix}.{e}.up_proj.weight"))
                source = Source("stack_gate_up", names)
            else:
                source = Source("stack_down", tuple(f"{prefix}.{e}.down_proj.weight" for e in range(n)))
            missing = [x for x in source.names if x not in sft]
            if missing:
                raise LayoutError(f"{name}: fine-tuned checkpoint lacks {missing[:3]}")
        elif name in sft:
            source = Source("sft", (name,))
        elif name.startswith(base_only_prefixes):
            source = Source("base", (name,))
        else:
            raise LayoutError(f"no source for base tensor {name}")
        plan[name] = source
        if source.kind != "base":
            used.update(source.names)
    return plan, sft - used


def _header(path: Path) -> dict:
    with open(path, "rb") as fh:
        (size,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(size))


def _index(root: Path) -> dict[str, str]:
    return json.loads((root / INDEX_FILE).read_text())["weight_map"]


def fuse(base: Path, sft: Path, out: Path, *, shard_bytes: int = SHARD_BYTES,
         base_only_prefixes: tuple[str, ...] = ("model.visual.",)) -> int:
    """Write the fused checkpoint. Returns the number of tensors written."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    dtypes = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}
    base_index, sft_index = _index(base), _index(sft)
    headers = {shard: _header(base / shard) for shard in set(base_index.values())}
    plan, unused = plan_layout(base_index, sft_index, base_only_prefixes=base_only_prefixes)
    if unused:
        print(f"warning: {len(unused)} fine-tuned tensors are not used, e.g. {sorted(unused)[:3]}")

    handles: dict[Path, object] = {}

    def read(root: Path, index: dict[str, str], name: str):
        shard = root / index[name]
        if shard not in handles:
            handles[shard] = safe_open(str(shard), framework="pt")
        return handles[shard].get_tensor(name)

    def build(name: str):
        source = plan[name]
        if source.kind == "base":
            return read(base, base_index, name)
        tensors = [read(sft, sft_index, n) for n in source.names]
        if source.kind == "sft":
            return tensors[0]
        if source.kind == "stack_gate_up":
            return torch.stack([torch.cat(tensors[i:i + 2], dim=0) for i in range(0, len(tensors), 2)])
        return torch.stack(tensors)

    tmp = Path(f"{out}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    weight_map: dict[str, str] = {}
    buffer: dict[str, object] = {}
    buffered, shard_id = 0, 0

    def flush() -> None:
        nonlocal buffer, buffered, shard_id
        if buffer:
            filename = f"model-{shard_id:05d}.safetensors"
            save_file(buffer, str(tmp / filename))
            weight_map.update({k: filename for k in buffer})
            buffer, buffered, shard_id = {}, 0, shard_id + 1

    for name in sorted(plan):
        meta = headers[base_index[name]][name]
        tensor = build(name)
        if tuple(tensor.shape) != tuple(meta["shape"]):
            raise LayoutError(f"{name}: built shape {tuple(tensor.shape)}, base has {tuple(meta['shape'])}")
        want = dtypes[meta["dtype"]]
        if tensor.dtype != want:
            if tensor.dtype == torch.float32 and want == torch.bfloat16:
                tensor = tensor.to(torch.bfloat16)
            else:
                raise LayoutError(f"{name}: dtype {tensor.dtype}, base has {meta['dtype']}")
        buffer[name] = tensor.contiguous()
        buffered += tensor.numel() * tensor.element_size()
        if buffered >= shard_bytes:
            flush()
    flush()

    renames = {f"model-{i:05d}.safetensors": f"model-{i + 1:05d}-of-{shard_id:05d}.safetensors"
               for i in range(shard_id)}
    for old, new in renames.items():
        os.rename(tmp / old, tmp / new)
    weight_map = {k: renames[v] for k, v in weight_map.items()}
    (tmp / INDEX_FILE).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1))
    for item in sft.iterdir():
        if item.is_file() and item.suffix != ".safetensors" and item.name not in (INDEX_FILE, "config.json"):
            shutil.copy2(item, tmp / item.name)
    shutil.copy2(base / "config.json", tmp / "config.json")
    shutil.rmtree(out, ignore_errors=True)
    os.rename(tmp, out)
    return len(weight_map)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", type=Path, required=True, help="base Hugging Face checkpoint")
    parser.add_argument("--sft", type=Path, required=True, help="fine-tuned checkpoint to rewrite")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-only-prefix", action="append", default=None,
                        help="tensor-name prefix to take from the base (default: model.visual.)")
    args = parser.parse_args(argv)
    prefixes = tuple(args.base_only_prefix or ["model.visual."])
    n = fuse(args.base, args.sft, args.out, base_only_prefixes=prefixes)
    print(f"wrote {n} tensors to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
