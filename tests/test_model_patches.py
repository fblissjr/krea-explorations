"""Tests for the composable model-edge patch hook on build_graph / build_split_graph.

A patch is `(graph, model_ref) -> new_model_ref`: it inserts node(s) on the model edge and returns the new
ref. `model_node(class_type, **inputs)` is the generic factory. This is the single seam levers (a LoRA, an
attention bias, a residual steer) hang off, so the graph skeleton lives in one place instead of being
re-inlined per harness.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate import build_graph, build_split_graph, model_node, cond_node  # noqa: E402


def _g(**kw):
    return build_graph("p", unet="u", clip="c", vae="v", **kw)


def test_model_node_inserts_on_edge_and_returns_new_ref():
    g = {"ckpt": {"class_type": "UNETLoader", "inputs": {}}}
    ref = model_node("Krea2Foo", strength=2.0)(g, ["ckpt", 0])
    node = g[ref[0]]
    assert node["class_type"] == "Krea2Foo"
    assert node["inputs"]["model"] == ["ckpt", 0]          # wired the incoming edge into `model`
    assert node["inputs"]["strength"] == 2.0


def test_patches_chain_in_order_before_the_sampler():
    g = _g(model_patches=[model_node("NodeA"), model_node("NodeB")])
    last = g[g["sampler"]["inputs"]["model"][0]]
    assert last["class_type"] == "NodeB"                   # sampler reads the last patch
    assert g[last["inputs"]["model"][0]]["class_type"] == "NodeA"  # B chained after A


def test_patches_apply_after_the_lora():
    g = _g(lora="x.safetensors", model_patches=[model_node("NodeA")])
    a = next(v for v in g.values() if v["class_type"] == "NodeA")
    assert g[a["inputs"]["model"][0]]["class_type"] == "LoraLoaderModelOnly"


def test_no_patches_is_a_no_op():
    assert _g()["sampler"]["inputs"]["model"] == ["ckpt", 0]
    assert _g(model_patches=[]) == _g()                    # empty list changes nothing


def test_split_graph_takes_per_branch_patches():
    g = build_split_graph("p", unet_high="raw", unet_low="turbo", clip="c", vae="v", boundary=3, steps=8,
                          model_patches_high=[model_node("HighLever")],
                          model_patches_low=[model_node("LowLever")])
    adv = [v for v in g.values() if v["class_type"] == "KSamplerAdvanced"]
    high = next(v for v in adv if v["inputs"]["add_noise"] == "enable")
    low = next(v for v in adv if v["inputs"]["add_noise"] == "disable")
    assert g[high["inputs"]["model"][0]]["class_type"] == "HighLever"
    assert g[low["inputs"]["model"][0]]["class_type"] == "LowLever"


# --- cond_patches: the conditioning-edge mirror of model_patches (for concept-inject-style levers) ---

def test_cond_node_inserts_on_the_conditioning_edge():
    g = {"pos": {"class_type": "CLIPTextEncode", "inputs": {}}}
    ref = cond_node("Krea2ConceptInject", scale=2.0, mode="amplify")(g, ["pos", 0])
    node = g[ref[0]]
    assert node["class_type"] == "Krea2ConceptInject"
    assert node["inputs"]["conditioning"] == ["pos", 0]    # wired the incoming cond edge into `conditioning`
    assert node["inputs"]["scale"] == 2.0


def test_cond_patches_wrap_the_positive_and_chain_in_order():
    g = _g(cond_patches=[cond_node("Krea2ConceptInject", scale=1.0), cond_node("Krea2ConceptInject", scale=2.0)])
    pos = g["sampler"]["inputs"]["positive"]
    assert g[pos[0]]["inputs"]["scale"] == 2.0                                  # sampler reads the last cond patch
    first = g[pos[0]]["inputs"]["conditioning"]
    assert g[first[0]]["inputs"]["scale"] == 1.0                                # second chained after first
    assert g[first[0]]["inputs"]["conditioning"] == ["pos", 0]                  # first wraps the text encode
    assert g["sampler"]["inputs"]["negative"] == ["neg", 0]                     # negative untouched


def test_no_cond_patches_is_a_no_op():
    assert _g()["sampler"]["inputs"]["positive"] == ["pos", 0]
    assert _g(cond_patches=[]) == _g()
