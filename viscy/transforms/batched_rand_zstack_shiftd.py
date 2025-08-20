import torch
from monai.transforms import MapTransform, RandomizableTransform
from torch import Tensor
from typing_extensions import Iterable


class BatchedRandZStackShiftd(MapTransform, RandomizableTransform):
    """Batched random Z-axis shifts for 3D microscopy data."""

    def __init__(
        self,
        keys: str | Iterable[str],
        max_shift: int = 3,
        prob: float = 0.1,
        mode: str = "constant",
        cval: float = 0.0,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.max_shift = max_shift
        self.mode = mode
        self.cval = cval

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        self.randomize(None)
        d = dict(sample)

        for key in self.key_iterator(d):
            data = d[key]
            if self.R.rand() < self.prob:
                batch_size, channels, depth, height, width = data.shape

                # Generate random shifts for the batch
                shifts = torch.randint(
                    -self.max_shift,
                    self.max_shift + 1,
                    (batch_size,),
                    device=data.device,
                )

                # Process samples with shifts
                result = data.clone()
                for b in range(batch_size):
                    shift = shifts[b].item()
                    if shift != 0:
                        if shift > 0:
                            # Shift down, pad at top
                            result[b, :, :shift] = self.cval
                            result[b, :, shift:] = data[b, :, :-shift]
                        else:
                            # Shift up, pad at bottom
                            shift = -shift
                            result[b, :, :-shift] = data[b, :, shift:]
                            result[b, :, -shift:] = self.cval

                d[key] = result

        return d
