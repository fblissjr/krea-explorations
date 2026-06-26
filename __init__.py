"""ComfyUI custom-node entry point for krea2-explorations.

ComfyUI imports this file from custom_nodes/krea2-explorations/. The actual toolkit is the
``krea2_explorations`` package under ``src/``; we add it to ``sys.path`` so the node works whether or
not the package is installed in ComfyUI's environment.
"""

import os
import sys

_SRC = os.path.join(os.path.dirname(__file__), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from krea2_explorations.comfy_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
