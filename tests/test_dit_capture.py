"""Logic tests for the EXP5 DiT-capture pooling/reduction helpers (numpy-only; the node's comfy plumbing is
smoke-tested at runtime). The COND row is the LAST batch row (CFG batches as [uncond, cond]; unbatched cond
pass is a single row) -- the 2026-06-28 review caught that pooling [0] captured uncond."""

import numpy as np

from krea2_explorations.krea2_dit_capture_node import (
    image_residual_spatial,
    image_residual_summary,
    reduce_attn_to_text,
)


def test_image_residual_summary_pools_image_tail_cond_row():
    out = np.arange(2 * 5 * 3, dtype=float).reshape(2, 5, 3)  # (B=2, seq=5, feat=3); row1 = cond
    s = image_residual_summary(out, txtlen=2)  # image tokens = indices 2,3,4
    assert s.shape == (3,)
    assert np.allclose(s, out[-1, 2:, :].mean(0))  # LAST (cond) row, mean over image tokens


def test_image_residual_summary_txtlen_zero():
    out = np.ones((1, 4, 2), dtype=float)
    s = image_residual_summary(out, 0)
    assert s.shape == (2,) and np.allclose(s, 1.0)


def test_image_residual_spatial_per_token_norm_cond_row():
    out = np.zeros((2, 4, 3), dtype=float)
    out[-1, 2] = [3.0, 4.0, 0.0]  # one image token (idx 2), cond row -> norm 5
    out[-1, 3] = [0.0, 0.0, 0.0]
    sp = image_residual_spatial(out, txtlen=2)  # image tokens 2,3
    assert sp.shape == (2,)
    assert np.allclose(sp, [5.0, 0.0])


def test_reduce_attn_to_text_averages_over_image_queries_and_heads():
    # (H=2, n_img=3, n_keys=5), txtlen=2 -> per-text-key mass averaged over queries+heads
    attn = np.zeros((2, 3, 5), dtype=float)
    attn[..., 0] = 0.5  # all image queries put 0.5 on text token 0
    attn[..., 1] = 0.1  # and 0.1 on text token 1
    per_txt = reduce_attn_to_text(attn, txtlen=2)
    assert per_txt.shape == (2,)
    assert np.allclose(per_txt, [0.5, 0.1])
