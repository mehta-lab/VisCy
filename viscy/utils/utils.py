"""General utility functions."""

import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from viscy.utils.pylogger import RankedLogger
from viscy.utils.rich_utils import enforce_tags, print_config_tree

log = RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Apply optional utilities before the task is started.

    Utilities:
    - Ignore python warnings
    - Set tags from command line
    - Pretty print config tree

    Parameters
    ----------
    cfg : DictConfig
        Configuration composed by Hydra.
    """
    # Return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # Disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # Prompt user to input tags from command line if they are not specified in config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        enforce_tags(cfg, save_to_file=True)

    # Pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        print_config_tree(cfg, resolve=False, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Wrapper that handles task execution with proper cleanup.

    This wrapper:
    - Executes the task function
    - Handles exceptions gracefully
    - Ensures cleanup (close loggers, print output dir)
    - Prevents multirun failures

    Parameters
    ----------
    task_func : Callable
        The task function to wrap.

    Returns
    -------
    Callable
        Wrapped task function.
    """

    def wrap(cfg: DictConfig) -> tuple[Dict[str, Any], Dict[str, Any]]:
        # Execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # Graceful exception handling
        except Exception as ex:
            log.exception("")  # Save exception to `.log` file
            raise ex

        # Cleanup and logging
        finally:
            path = HydraConfig.get().runtime.output_dir
            log.info(f"Output dir: {path}")

            # Close wandb if it was used
            if find_spec("wandb"):  # Check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(
    metric_dict: Dict[str, Any], metric_name: str | None
) -> float | None:
    """Safely retrieve metric value from a metric dictionary.

    Parameters
    ----------
    metric_dict : Dict[str, Any]
        Dictionary with metric values.
    metric_name : str | None
        If provided, name of the metric to retrieve.

    Returns
    -------
    float | None
        Metric value if found, None otherwise.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value
