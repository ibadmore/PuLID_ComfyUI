import torch
from .pulid_model import PuLIDModel

class PuLIDNode:
    def __init__(self):
        self.model_cache = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (["xl", "flux-bf16", "flux-fp8"],),
                "method": (["fidelity", "style", "neutral"],),
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "fidelity": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "projection": (["ortho", "ortho_v2"],),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_pulid"

    CATEGORY = "conditioning/pulid"

    def apply_pulid(self, image, model, method, weight, fidelity=8.0, projection="ortho_v2"):
        if model not in self.model_cache:
            if model == "xl":
                self.model_cache[model] = PuLIDModel.load_xl()
            elif model == "flux-bf16":
                self.model_cache[model] = PuLIDModel.load_flux(fp8=False)
            elif model == "flux-fp8":
                self.model_cache[model] = PuLIDModel.load_flux(fp8=True)
        
        pulid_model = self.model_cache[model]
        conditioning = pulid_model.process(image, method, weight, fidelity, projection)
        
        return (conditioning,)

# ... (rest of the file remains unchanged)