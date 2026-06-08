from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from experiments.idm.model.idm import FeatureIDM, PatchTransformerIDM


class ImageIDMAdapter:
    def __init__(self, checkpoint_path: str | Path, encoder_name: str, device: torch.device) -> None:
        self.device = device
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        args = checkpoint.get("args", {})
        self.model_type = str(args.get("model_type", "feature"))
        self.action_width = int(args.get("action_width", 7))
        self.horizon = int(args.get("eval_action_prefix") or int(checkpoint["action_dim"]) // self.action_width)

        if self.model_type == "patch":
            self.model = PatchTransformerIDM(
                feature_dim=int(checkpoint["feature_dim"]),
                action_width=self.action_width,
                horizon=self.horizon,
                proprio_dim=int(checkpoint["proprio_dim"]),
                width=int(args.get("hidden_dim", 512)),
                depth=int(args.get("depth", 4)),
                heads=int(args.get("heads", 8)),
                mlp_ratio=float(args.get("mlp_ratio", 4.0)),
                dropout=float(args.get("dropout", 0.1)),
            ).to(device)
        else:
            self.model = FeatureIDM(
                feature_dim=int(checkpoint["feature_dim"]),
                action_dim=int(checkpoint["action_dim"]),
                proprio_dim=int(checkpoint["proprio_dim"]),
                hidden_dim=int(args.get("hidden_dim", 512)),
                depth=int(args.get("depth", 4)),
            ).to(device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name).to(device).eval()

    @property
    def adapter_note(self) -> str:
        if self.model_type != "patch":
            return "Feature IDM uses pooled C/P features, matching old eval input."
        return "Patch IDM adapts online C/P pairs to the training window as [C repeated k times, P]."

    def _pil(self, image: Image.Image | np.ndarray) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")

    def _encode_pooled(self, images: list[Image.Image]) -> torch.Tensor:
        batch = self.processor(images=images, return_tensors="pt")
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.no_grad():
            if hasattr(self.encoder, "get_image_features"):
                output = self.encoder.get_image_features(**batch)
            else:
                output = self.encoder(**batch)
        if isinstance(output, torch.Tensor):
            features = output
        elif getattr(output, "pooler_output", None) is not None:
            features = output.pooler_output
        elif hasattr(output, "last_hidden_state"):
            features = output.last_hidden_state.mean(dim=1)
        else:
            raise RuntimeError("Encoder output has neither pooler_output nor last_hidden_state")
        return torch.nn.functional.normalize(features.float(), dim=-1)

    def _encode_patches(self, images: list[Image.Image]) -> torch.Tensor:
        batch = self.processor(images=images, return_tensors="pt")
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.no_grad():
            vision_model = getattr(self.encoder, "vision_model", self.encoder)
            output = vision_model(**batch)
        if not hasattr(output, "last_hidden_state"):
            raise RuntimeError("Patch IDM requires encoder last_hidden_state patch tokens")
        return output.last_hidden_state.float()

    def _patch_window(self, z_current: torch.Tensor, z_future: torch.Tensor) -> torch.Tensor:
        current = z_current.unsqueeze(1).expand(-1, self.horizon, -1, -1)
        future = z_future.unsqueeze(1)
        return torch.cat([current, future], dim=1)

    def predict(
        self,
        current_images: list[Image.Image | np.ndarray],
        future_images: list[Image.Image | np.ndarray],
        proprio: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        current = [self._pil(image) for image in current_images]
        future = [self._pil(image) for image in future_images]
        proprio_t = torch.as_tensor(proprio, dtype=torch.float32, device=self.device).reshape(len(current), -1)
        with torch.no_grad():
            if self.model_type == "patch":
                z_current = self._encode_patches(current)
                z_future = self._encode_patches(future)
                pred = self.model(self._patch_window(z_current, z_future), proprio_t)
            else:
                z_current = self._encode_pooled(current)
                z_future = self._encode_pooled(future)
                pred = self.model(z_current, z_future, proprio_t)
        return pred.detach().cpu().numpy().astype(np.float32).reshape(len(current), self.horizon, self.action_width)

    def feature_similarity(
        self,
        images_a: list[Image.Image | np.ndarray],
        images_b: list[Image.Image | np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        a = self._encode_pooled([self._pil(image) for image in images_a])
        b = self._encode_pooled([self._pil(image) for image in images_b])
        delta_l2 = torch.linalg.norm(a - b, dim=-1).detach().cpu().numpy().astype(np.float32)
        cosine = torch.nn.functional.cosine_similarity(a, b, dim=-1).detach().cpu().numpy().astype(np.float32)
        return delta_l2, cosine
