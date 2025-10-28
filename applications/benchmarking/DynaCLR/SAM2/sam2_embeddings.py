import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional

import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from skimage.exposure import rescale_intensity

sys.path.append(str(Path(__file__).parent.parent))

from base_embedding_module import BaseEmbeddingModule, create_embedding_cli


class SAM2Module(BaseEmbeddingModule):
    def __init__(
        self,
        model_name: str = "facebook/sam2-hiera-base-plus",
        channel_reduction_methods: Optional[
            Dict[str, Literal["middle_slice", "mean", "max"]]
        ] = None,
        channel_names: Optional[List[str]] = None,
        middle_slice_index: Optional[int] = None,
    ):
        super().__init__(channel_reduction_methods, channel_names, middle_slice_index)
        self.model_name = model_name
        self.model = None  # Initialize in on_predict_start when device is set

    @classmethod
    def from_config(cls, cfg):
        """Create model instance from configuration."""
        model_config = cfg.get("model", {})

        return cls(
            model_name=model_config.get("model_name", "facebook/sam2-hiera-base-plus"),
            channel_reduction_methods=model_config.get("channel_reduction_methods", {}),
            middle_slice_index=model_config.get("middle_slice_index", None),
        )

    def on_predict_start(self):
        """Initialize model with proper device when prediction starts."""
        if self.model is None:
            self.model = SAM2ImagePredictor.from_pretrained(
                self.model_name, device=self.device
            )

    def _process_input(self, x: torch.Tensor):
        """Convert input tensor to 3-channel RGB format as needed for SAM2."""
        return self._convert_to_rgb(x)

    def _extract_features(self, image_list):
        """Extract features using SAM2 model."""
        self.model.set_image_batch(image_list)
        # Extract high-resolution features and apply global average pooling
        features = self.model._features["high_res_feats"][0].mean(dim=(2, 3))
        return features

    def _convert_to_rgb(self, x: torch.Tensor) -> list:
        """
        Convert input tensor to 3-channel RGB format as needed for SAM2.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with 1, 2, or 3+ channels and shape (B, C, H, W).

        Returns
        -------
        list of numpy.ndarray
            List of numpy arrays in HWC format for SAM2 processing.
        """
        # Convert to RGB and scale to [0, 255] range for SAM2
        if x.shape[1] == 1:
            x_rgb = x.repeat(1, 3, 1, 1) * 255.0
        elif x.shape[1] == 2:
            x_3ch = torch.zeros(
                (x.shape[0], 3, x.shape[2], x.shape[3]), device=x.device
            )
            x[:, 0] = rescale_intensity(x[:, 0], out_range="uint8")
            x[:, 1] = rescale_intensity(x[:, 1], out_range="uint8")

            x_3ch[:, 0] = x[:, 0]
            x_3ch[:, 1] = x[:, 1]
            x_3ch[:, 2] = 0.5 * (x[:, 0] + x[:, 1])  # B channel as blend
            x_rgb = x_3ch

        elif x.shape[1] == 3:
            x_rgb = rescale_intensity(x, out_range="uint8")
        else:
            # More than 3 channels, normalize first 3 and scale
            x_3ch = x[:, :3]
            x_rgb = rescale_intensity(x_3ch, out_range="uint8")

        # Convert to list of numpy arrays in HWC format for SAM2
        return [
            x_rgb[i].cpu().numpy().transpose(1, 2, 0) for i in range(x_rgb.shape[0])
        ]


if __name__ == "__main__":
    main = create_embedding_cli(SAM2Module, "SAM2")
    main()
