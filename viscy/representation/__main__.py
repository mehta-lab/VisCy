from viscy.cli import VisCyCLI, run_cli
from viscy.data.hcs import HCSDataModule
from viscy.trainer import VisCyTrainer
from viscy.translation.engine import VSUNet

if __name__ == "__main__":
    run_cli(
        cli_class=VisCyCLI,
        model_class=VSUNet,
        datamodule_class=HCSDataModule,
        trainer_class=VisCyTrainer,
    )
