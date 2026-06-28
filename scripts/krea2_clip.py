"""Load the Krea 2 CLIP on CPU and pool its conditioning -- shared helper for the direction/analysis scripts.

ComfyUI's ``comfy`` package lives in the ComfyUI repo (not pip-installed), so this boots it in CPU mode
before importing. Use it to turn a prompt into the pooled 12-layer conditioning vector that
``krea2_explorations.contrast_directions.pooled_direction`` consumes.
"""

import os
import sys
from pathlib import Path

# parents[3] = the ComfyUI install root (this custom-node repo lives at <comfyui>/custom_nodes/<repo>/)
DEFAULT_COMFY_ROOT = Path(__file__).resolve().parents[3]


def load_clip_cpu(comfy_root=DEFAULT_COMFY_ROOT, text_encoder="qwen3vl_4b_bf16.safetensors"):
    """Boot ComfyUI in CPU mode and load the Krea 2 CLIP (``CLIPType.KREA2``). Returns the comfy CLIP object."""
    comfy_root = Path(comfy_root)
    sys.path.insert(0, str(comfy_root))
    os.chdir(comfy_root)
    sys.argv = [sys.argv[0], "--cpu"]  # comfy's own CPU flag (set before enable_args_parsing)
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.sd
    te = comfy_root / "models" / "text_encoders" / text_encoder
    return comfy.sd.load_clip(ckpt_paths=[str(te)], clip_type=comfy.sd.CLIPType.KREA2)


def pooled_conditioning(clip, prompt):
    """Encode a prompt and mean-pool the (post-strip) 12-layer conditioning over tokens -> ``(12*2560,)`` array."""
    cond = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))[0][0].float()  # (1, seq, 12*2560)
    return cond.mean(dim=1)[0].cpu().numpy()
