from monai.data import decollate_batch
from monai.transforms import Transform
from torch import Tensor


class Decollate(Transform):
    def __init__(
        self, detach: bool = True, pad_batch: bool = True, fill_value=None
    ) -> None:
        super().__init__()
        self.detach = detach
        self.pad_batch = pad_batch
        self.fill_value = fill_value

    def __call__(self, data: Tensor) -> Tensor:
        return decollate_batch(
            batch=data,
            detach=self.detach,
            pad=self.pad_batch,
            fill_value=self.fill_value,
        )
