from typing import Dict, Tuple, Optional
import numpy as np

class CoTVLAWrapper:
    """
    Minimal wrapper for a CoT‑VLA style model that produces:
      • heatmap (H×W) of target likelihood
      • optional action token(s) (text) describing the immediate action
    Replace `infer()` with your actual model forward pass.
    """
    def __init__(self, model_name: str = "cot_vla_stub", device: str = "cpu"):
        self.model_name = model_name
        self.device = device

    def warmup(self, shape: Tuple[int, int] = (480, 640)):
        _ = np.zeros(shape, dtype=np.float32)
        return True

    def infer(
        self,
        rgb: np.ndarray,                # H×W×3 uint8 BGR/RGB
        instruction: str,
        context: Optional[Dict] = None,
    ) -> Dict:
        H, W = rgb.shape[:2]
        # --- Dummy heatmap: center‑biased Gaussian (replace with real output) ---
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        cy, cx = int(H*0.55), int(W*0.55)
        heat = np.exp(-(((yy - cy)**2)/ (2*(0.08*H)**2) + ((xx - cx)**2)/ (2*(0.08*W)**2)))
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
        action = "grasp_nozzle" if "노즐" in instruction or "nozzle" in instruction.lower() else "move_to_target"
        return {"heatmap": heat.astype(np.float32), "action": action}

