import pytest

from autoenvscaling.training.fuse_experts import LayoutError, Source, plan_layout

PRE = "model.language_model.layers.0.mlp.experts"
MTP = "mtp.layers.0.mlp.experts"


def sft_experts(prefix: str, n: int) -> list[str]:
    return [f"{prefix}.{e}.{p}.weight" for e in range(n) for p in ("gate_proj", "up_proj", "down_proj")]


def test_plan_stacks_experts_in_numeric_order_and_keeps_base_only_tensors():
    base = [f"{PRE}.gate_up_proj", f"{PRE}.down_proj", f"{MTP}.gate_up_proj", f"{MTP}.down_proj",
            "model.embed_tokens.weight", "model.visual.blocks.0.attn.qkv.weight"]
    sft = sft_experts(PRE, 12) + sft_experts(MTP, 2) + ["model.embed_tokens.weight", "extra.weight"]
    plan, unused = plan_layout(base, sft)
    gate_up = plan[f"{PRE}.gate_up_proj"]
    assert gate_up.kind == "stack_gate_up"
    assert gate_up.names[:4] == (f"{PRE}.0.gate_proj.weight", f"{PRE}.0.up_proj.weight",
                                 f"{PRE}.1.gate_proj.weight", f"{PRE}.1.up_proj.weight")
    assert gate_up.names[-2] == f"{PRE}.11.gate_proj.weight"       # 11 after 10, not after 1
    assert plan[f"{MTP}.down_proj"] == Source("stack_down", (f"{MTP}.0.down_proj.weight",
                                                             f"{MTP}.1.down_proj.weight"))
    assert plan["model.embed_tokens.weight"] == Source("sft", ("model.embed_tokens.weight",))
    assert plan["model.visual.blocks.0.attn.qkv.weight"].kind == "base"
    assert unused == {"extra.weight"}


def test_plan_copies_already_stacked_experts():
    plan, _ = plan_layout([f"{PRE}.gate_up_proj"], [f"{PRE}.gate_up_proj"])
    assert plan[f"{PRE}.gate_up_proj"].kind == "sft"


def test_plan_errors():
    with pytest.raises(LayoutError, match="not contiguous"):
        plan_layout([f"{PRE}.down_proj"], [f"{PRE}.0.gate_proj.weight", f"{PRE}.2.gate_proj.weight"])
    with pytest.raises(LayoutError, match="lacks"):
        plan_layout([f"{PRE}.down_proj"], [f"{PRE}.0.gate_proj.weight"])
    with pytest.raises(LayoutError, match="no source"):
        plan_layout(["model.norm.weight"], [])
