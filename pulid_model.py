import torch
from transformers import AutoProcessor, CLIPVisionModelWithProjection
import os

class PuLIDModel:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    @classmethod
    def load_xl(cls):
        # Implement XL model loading here
        raise NotImplementedError("XL model loading not yet implemented")

    @classmethod
    def load_flux(cls, fp8=False):
        model_path = os.path.join("models", "pulid", "PuLID-FLUX-v0.9.0")
        try:
            if fp8:
                model = CLIPVisionModelWithProjection.from_pretrained(
                    model_path,
                    torch_dtype=torch.float8_e4m3fn,
                    device_map="auto"
                )
            else:
                model = CLIPVisionModelWithProjection.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
            
            processor = AutoProcessor.from_pretrained(model_path)
            
            return cls(model, processor)
        except Exception as e:
            raise RuntimeError(f"Failed to load FLUX model: {str(e)}")

    def process(self, image, method, weight, fidelity=8.0, projection="ortho_v2"):
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            image_embeds = outputs.image_embeds
            
            if method == "fidelity":
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            elif method == "style":
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True) * 2

            if projection == "ortho":
                image_embeds = image_embeds - (image_embeds @ self.model.vision_model.embeddings.position_embedding.weight.T) @ self.model.vision_model.embeddings.position_embedding.weight
            elif projection == "ortho_v2":
                image_embeds = image_embeds - (image_embeds @ self.model.vision_model.embeddings.position_embedding.weight.T) @ self.model.vision_model.embeddings.position_embedding.weight / 2

            image_embeds = image_embeds * (fidelity / 8.0) * weight
            
            return image_embeds
        except Exception as e:
            raise RuntimeError(f"Error processing image: {str(e)}")

# ... (additional helper methods as needed)