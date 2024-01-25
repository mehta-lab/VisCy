from typing import Literal, Sequence

from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader

_MODES = Literal["min_size", "max_size_cycle", "max_size", "sequential"]


class CombinedDataModule(LightningDataModule):
    """Wrapper for combining multiple data modules.
    For supported modes, see ``lightning.pytorch.utilities.combined_loader``.

    :param Sequence[LightningDataModule] data_modules: data modules to combine
    :param str train_mode: mode in training stage, defaults to "max_size_cycle"
    :param str val_mode: mode in validation stage, defaults to "sequential"
    :param str test_mode: mode in testing stage, defaults to "sequential"
    :param str predict_mode: mode in prediction stage, defaults to "sequential"
    """

    def __init__(
        self,
        data_modules: Sequence[LightningDataModule],
        train_mode: _MODES = "max_size_cycle",
        val_mode: _MODES = "sequential",
        test_mode: _MODES = "sequential",
        predict_mode: _MODES = "sequential",
    ):
        super().__init__()
        self.data_modules = data_modules
        self.train_mode = train_mode
        self.val_mode = val_mode
        self.test_mode = test_mode
        self.predict_mode = predict_mode

    def prepare_data(self):
        for dm in self.data_modules:
            dm.prepare_data()

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]):
        for dm in self.data_modules:
            dm.setup(stage)

    def train_dataloader(self):
        return CombinedLoader(
            [dm.train_dataloader() for dm in self.data_modules], mode=self.train_mode
        )

    def val_dataloader(self):
        return CombinedLoader(
            [dm.val_dataloader() for dm in self.data_modules], mode=self.val_mode
        )

    def test_dataloader(self):
        return CombinedLoader(
            [dm.test_dataloader() for dm in self.data_modules], mode=self.test_mode
        )

    def predict_dataloader(self):
        return CombinedLoader(
            [dm.predict_dataloader() for dm in self.data_modules],
            mode=self.predict_mode,
        )
