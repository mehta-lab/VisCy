import torch
from monai.transforms import MapTransform, RandomizableTransform
from torch import Tensor
from typing_extensions import Iterable


class BatchedRandHistogramShiftd(MapTransform, RandomizableTransform):
    """

    Apply random histogram shifts to modify intensity distributions.

    Parameters
    ----------
    keys : str or Iterable[str]
        Keys of the corresponding items to be transformed.
    shift_range : tuple[float, float], optional
        Range for random intensity shift values, by default (-0.1, 0.1).
    prob : float, optional
        Probability of applying the transform, by default 0.1.
    allow_missing_keys : bool, optional
        Whether to ignore missing keys, by default False.
    """

    def __init__(
        self,
        keys: str | Iterable[str],
        shift_range: tuple[float, float] = (-0.1, 0.1),
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.shift_range = shift_range

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """Apply histogram shift to sample data.

        Parameters
        ----------
        sample : dict[str, Tensor]
            Dictionary containing image tensors to transform.

        Returns
        -------
        dict[str, Tensor]
            Dictionary with intensity-shifted tensors.
        """
        self.randomize(None)
        d = dict(sample)

        for key in self.key_iterator(d):
            data = d[key]
            if self.R.rand() < self.prob:
                batch_size = data.shape[0]

                # Generate random shifts for the batch
                shifts = torch.empty(batch_size, device=data.device, dtype=data.dtype)
                shift_min, shift_max = self.shift_range
                shifts.uniform_(shift_min, shift_max)

                # Apply shifts to batch
                shifts = shifts.view(batch_size, 1, 1, 1, 1)
                d[key] = data + shifts

        return d
