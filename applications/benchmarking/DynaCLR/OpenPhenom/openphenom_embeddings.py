import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional

import torch
from transformers import AutoModel

sys.path.append(str(Path(__file__).parent.parent))

from base_embedding_module import BaseEmbeddingModule, create_embedding_cli


class OpenPhenomModule(BaseEmbeddingModule):
    def __init__(
        self,
        channel_reduction_methods: Optional[
            Dict[str, Literal["middle_slice", "mean", "max"]]
        ] = None,
        channel_names: Optional[List[str]] = None,
        middle_slice_index: Optional[int] = None,
    ):
        super().__init__(channel_reduction_methods, channel_names, middle_slice_index)

        try:
            self.model = AutoModel.from_pretrained(
                "recursionpharma/OpenPhenom", trust_remote_code=True
            )
            self.model.eval()
        except ImportError:
            raise ImportError(
                "Please install the OpenPhenom dependencies: pip install transformers"
            )

    @classmethod
    def from_config(cls, cfg):
        """Create model instance from configuration."""
        model_config = cfg.get("model", {})
        dm_config = cfg.get("datamodule", {})

        return cls(
            channel_reduction_methods=model_config.get("channel_reduction_methods", {}),
            channel_names=dm_config.get("source_channel", []),
        )

    def on_predict_start(self):
        """Move model to GPU when prediction starts."""
        self.model.to(self.device)

    def _process_input(self, x: torch.Tensor):
        """Convert to uint8 as OpenPhenom expects uint8 inputs."""
        if x.dtype != torch.uint8:
            x = (
                ((x - x.min()) / (x.max() - x.min()) * 255)
                .clamp(0, 255)
                .to(torch.uint8)
            )
        return x

    def _extract_features(self, processed_input):
        """Extract features using OpenPhenom model."""
        # Get embeddings
        self.model.return_channelwise_embeddings = False
        features = self.model.predict(processed_input)
        return features


if __name__ == "__main__":
    main = create_embedding_cli(OpenPhenomModule, "OpenPhenom")
    main()
