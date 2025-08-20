from monai.transforms import MapTransform, RandomizableTransform
from torch import Tensor
from typing_extensions import Iterable
import torch


class BatchedRandFlipd(MapTransform, RandomizableTransform):
    """Apply random flips to batched data.
    
    This transform applies random flips along specified spatial axes to batched data
    with shape [B, C, D, H, W].
    """
    
    def __init__(self, keys, spatial_axis=[0, 1, 2], prob=0.5, allow_missing_keys=False):
        """
        Args:
            keys: Keys to apply flipping to
            spatial_axis: List of spatial axes to randomly flip (0=D, 1=H, 2=W)
            prob: Probability of applying each flip
            allow_missing_keys: Whether to allow missing keys
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.spatial_axis = spatial_axis
        
    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        self.randomize(None)
        d = dict(sample)
        
        for key in self.key_iterator(d):
            data = d[key]  # Shape: [B, C, D, H, W]
            
            # Apply flips for each spatial axis independently
            for axis in self.spatial_axis:
                if self.R.rand() < self.prob:
                    # Flip along spatial axis (add 2 to account for batch and channel dims)
                    flip_axis = axis + 2
                    data = torch.flip(data, dims=[flip_axis])
            
            d[key] = data
            
        return d