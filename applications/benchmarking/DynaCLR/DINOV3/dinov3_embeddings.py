import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
from PIL import Image
from skimage.exposure import rescale_intensity
from transformers import AutoImageProcessor, AutoModel

sys.path.append(str(Path(__file__).parent.parent))

from base_embedding_module import BaseEmbeddingModule, create_embedding_cli


class DINOv3Module(BaseEmbeddingModule):
    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        channel_reduction_methods: Optional[
            Dict[str, Literal["middle_slice", "mean", "max"]]
        ] = None,
        channel_names: Optional[List[str]] = None,
        pooling_method: Literal["mean", "max", "cls_token"] = "mean",
        middle_slice_index: Optional[int] = None,
    ):
        super().__init__(channel_reduction_methods, channel_names, middle_slice_index)
        self.model_name = model_name
        self.pooling_method = pooling_method

        self.model = None
        self.processor = None

    @classmethod
    def from_config(cls, cfg):
        """Create model instance from configuration."""
        model_config = cfg.get("model", {})
        return cls(
            model_name=model_config.get(
                "model_name", "facebook/dinov3-vitb16-pretrain-lvd1689m"
            ),
            pooling_method=model_config.get("pooling_method", "mean"),
            channel_reduction_methods=model_config.get("channel_reduction_methods", {}),
            channel_names=model_config.get("channel_names", []),
            middle_slice_index=model_config.get("middle_slice_index", None),
        )

    def on_predict_start(self):
        if self.model is None:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            self.model.to(self.device)

    def _process_input(self, x: torch.Tensor):
        """Convert tensor to PIL Images for DINOv3 processing."""
        return self._convert_to_pil_images(x)

    def _extract_features(self, pil_images):
        """Extract features using DINOv3 model."""
        inputs = self.processor(pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            token_features = outputs.last_hidden_state
            features = self._pool_features(token_features)

        return features

    def _convert_to_pil_images(self, x: torch.Tensor) -> List[Image.Image]:
        """
        Convert tensor to list of PIL Images for DINOv3 processing.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (B, C, H, W).

        Returns
        -------
        list of PIL.Image.Image
            List of PIL Images ready for DINOv3 processing.
        """
        images = []

        for b in range(x.shape[0]):
            img_tensor = x[b]  # (C, H, W)

            if img_tensor.shape[0] == 1:
                # Single channel - convert to grayscale PIL
                img_array = img_tensor[0].cpu().numpy()
                # Normalize to 0-255
                img_normalized = (
                    (img_array - img_array.min())
                    / (img_array.max() - img_array.min())
                    * 255
                ).astype(np.uint8)
                pil_img = Image.fromarray(img_normalized, mode="L")

            elif img_tensor.shape[0] == 2:
                img_array = img_tensor.cpu().numpy()
                rgb_array = np.zeros(
                    (img_array.shape[1], img_array.shape[2], 3), dtype=np.uint8
                )

                ch0_norm = rescale_intensity(img_array[0], out_range=(0, 255)).astype(
                    np.uint8
                )
                ch1_norm = rescale_intensity(img_array[1], out_range=(0, 255)).astype(
                    np.uint8
                )

                rgb_array[:, :, 0] = ch0_norm  # Red
                rgb_array[:, :, 1] = ch1_norm  # Green
                rgb_array[:, :, 2] = (ch0_norm + ch1_norm) // 2  # Blue

                pil_img = Image.fromarray(rgb_array, mode="RGB")

            elif img_tensor.shape[0] == 3:
                # Three channels - direct RGB
                img_array = img_tensor.cpu().numpy().transpose(1, 2, 0)  # HWC
                img_normalized = rescale_intensity(
                    img_array, out_range=(0, 255)
                ).astype(np.uint8)
                pil_img = Image.fromarray(img_normalized, mode="RGB")

            else:
                # More than 3 channels - use first 3
                img_array = img_tensor[:3].cpu().numpy().transpose(1, 2, 0)  # HWC
                img_normalized = rescale_intensity(
                    img_array, out_range=(0, 255)
                ).astype(np.uint8)
                pil_img = Image.fromarray(img_normalized, mode="RGB")

            images.append(pil_img)

        return images

    def _pool_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Pool spatial features from DINOv3 tokens.

        Parameters
        ----------
        features : torch.Tensor
            Token features with shape (B, num_tokens, hidden_dim).

        Returns
        -------
        torch.Tensor
            Pooled features with shape (B, hidden_dim).
        """
        if self.pooling_method == "cls_token":
            # For ViT models, first token is usually CLS token
            if "vit" in self.model_name.lower():
                return features[:, 0, :]  # CLS token
            else:
                # For ConvNeXt, no CLS token, fall back to mean
                return features.mean(dim=1)

        elif self.pooling_method == "max":
            return features.max(dim=1)[0]
        else:  # mean pooling
            return features.mean(dim=1)


if __name__ == "__main__":
    main = create_embedding_cli(DINOv3Module, "DINOv3")
    main()
