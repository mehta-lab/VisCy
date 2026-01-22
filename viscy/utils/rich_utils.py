"""Rich utilities for pretty printing and user prompts."""

from pathlib import Path
from typing import Sequence

import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.prompt import Prompt

from viscy.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Print configuration tree using Rich library.

    Parameters
    ----------
    cfg : DictConfig
        Configuration composed by Hydra.
    print_order : Sequence[str]
        Order in which to print config sections.
    resolve : bool
        Whether to resolve interpolations.
    save_to_file : bool
        Whether to save to file.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # Add fields from `print_order` first
    for field in print_order:
        (
            queue.append(field)
            if field in cfg
            else log.warning(
                f"Field '{field}' not found in config. "
                f"Available fields: {', '.join(cfg.keys())}"
            )
        )

    # Add remaining fields not in `print_order`
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # Generate config tree
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # Print and save
    rich.print(tree)

    if save_to_file:
        output_dir = Path(HydraConfig.get().runtime.output_dir)
        with open(output_dir / "config_tree.log", "w") as file:
            rich.print(tree, file=file)


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompt user for tags if not specified in config.

    Parameters
    ----------
    cfg : DictConfig
        Configuration composed by Hydra.
    save_to_file : bool
        Whether to save tags to file.
    """
    if not cfg.get("tags"):
        if "id" in HydraConfig.get().cfg.hydra.job:
            raise ValueError("Specify tags before launching a multirun!")

        log.warning("No tags found in config. Prompting user for tags...")
        tags = Prompt.ask("Enter a list of tags", default="dev")
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

        log.info(f"Tags: {cfg.tags}")

    if save_to_file:
        output_dir = Path(HydraConfig.get().runtime.output_dir)
        with open(output_dir / "tags.log", "w") as file:
            rich.print(cfg.tags, file=file)
