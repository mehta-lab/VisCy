import sys

from viscy.cli import VisCyCLI, run_cli
from viscy.data.hcs import HCSDataModule
from viscy.trainer import VisCyTrainer
from viscy.translation.engine import VSUNet


class TranslationCLI(VisCyCLI):
    """Extending lightning CLI arguments and defualts."""

    def add_arguments_to_parser(self, parser) -> None:
        super().add_arguments_to_parser(parser)
        if "preprocess" not in sys.argv:
            parser.link_arguments("data.yx_patch_size", "model.example_input_yx_shape")
            parser.link_arguments("model.architecture", "data.architecture")


if __name__ == "__main__":
    run_cli(
        cli_class=TranslationCLI,
        model_class=VSUNet,
        datamodule_class=HCSDataModule,
        trainer_class=VisCyTrainer,
    )
