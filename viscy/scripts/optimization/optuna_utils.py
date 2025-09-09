import glob
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_tensorboard_metric(
    log_dir: str, metric_name: str = "loss/total/val", aggregation: str = "min"
) -> float:
    """
    Extract a metric from TensorBoard logs.

    Args:
        log_dir: Path to the directory containing TensorBoard logs
        metric_name: Name of the metric to extract (e.g., "loss/total/val")
        aggregation: How to aggregate the metric values ("min", "max", "last", "mean")

    Returns:
        The aggregated metric value, or float('inf') if extraction fails

    Examples:
        >>> extract_tensorboard_metric("./logs/version_1", "loss/total/val", "min")
        0.234567

        >>> extract_tensorboard_metric("./logs/version_1", "accuracy", "max")
        0.891234
    """
    try:
        # Find the events file
        events_files = list(Path(log_dir).glob("events.out.tfevents.*"))
        if not events_files:
            print(f"Warning: No events file found in {log_dir}")
            return float("inf")

        # Load TensorBoard data
        ea = EventAccumulator(str(events_files[0]))
        ea.Reload()

        # Extract metric
        if metric_name in ea.Tags()["scalars"]:
            values = np.array([scalar.value for scalar in ea.Scalars(metric_name)])

            if aggregation == "min":
                result = float(np.min(values))
            elif aggregation == "max":
                result = float(np.max(values))
            elif aggregation == "last":
                result = float(values[-1])
            elif aggregation == "mean":
                result = float(np.mean(values))
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

            print(f"Extracted {metric_name} ({aggregation}): {result:.6f}")
            return result
        else:
            print(f"Warning: Metric '{metric_name}' not found in {log_dir}")
            available_metrics = ea.Tags()["scalars"]
            print(f"Available metrics: {available_metrics}")
            return float("inf")

    except Exception as e:
        print(f"Error extracting {metric_name} from {log_dir}: {e}")
        return float("inf")


def modify_config(
    base_config_path: str,
    modifications: Dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    """
    Modify a YAML configuration file with new parameter values.

    Supports nested key modification using dot notation (e.g., "model.init_args.beta").

    Args:
        base_config_path: Path to the base configuration file
        modifications: Dictionary with nested keys to modify
            e.g., {"model.init_args.beta": 10, "trainer.max_epochs": 50}
        output_path: Where to save the modified config (if None, creates temp file)

    Returns:
        Path to the modified configuration file

    Examples:
        >>> modify_config("base.yml", {"model.init_args.lr": 1e-3}, "modified.yml")
        "modified.yml"

        >>> temp_path = modify_config("base.yml", {"trainer.max_epochs": 100})
        >>> # Returns path to temporary file
    """
    # Load base config
    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply modifications
    for key_path, value in modifications.items():
        keys = key_path.split(".")
        current = config

        # Navigate to the nested dictionary
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the final value
        current[keys[-1]] = value

    # Save modified config
    if output_path is None:
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False)
        output_path = temp_file.name
        temp_file.close()

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return output_path


def run_lightning_training(
    config_path: str,
    working_dir: str = ".",
    timeout: int = 3600,
    capture_output: bool = True,
) -> subprocess.CompletedProcess:
    """
    Run Lightning training with the given configuration.

    Args:
        config_path: Path to the configuration file
        working_dir: Working directory for the training process
        timeout: Timeout in seconds (default: 1 hour)
        capture_output: Whether to capture stdout/stderr

    Returns:
        CompletedProcess object with training results

    Examples:
        >>> result = run_lightning_training("config.yml", timeout=1800)
        >>> if result.returncode == 0:
        ...     print("Training completed successfully")
    """
    cmd = ["python", "-m", "viscy.cli.train", "fit", "--config", config_path]

    print(f"Running command: {' '.join(cmd)}")

    return subprocess.run(
        cmd, cwd=working_dir, capture_output=capture_output, text=True, timeout=timeout
    )


def suggest_hyperparameters(
    trial, param_config: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Suggest hyperparameters based on a configuration dictionary.

    Supports different parameter types with flexible configuration options.

    Args:
        trial: Optuna trial object
        param_config: Configuration for parameters with format:
            {
                "param_name": {
                    "type": "float" | "int" | "categorical",
                    "low": <value>,     # for float/int
                    "high": <value>,    # for float/int
                    "choices": [<values>],  # for categorical
                    "log": True/False,  # for float/int (optional)
                    "step": <value>     # for int (optional)
                }
            }

    Returns:
        Dictionary of suggested parameter values

    Examples:
        >>> param_config = {
        ...     "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        ...     "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
        ...     "epochs": {"type": "int", "low": 10, "high": 100, "step": 10}
        ... }
        >>> params = suggest_hyperparameters(trial, param_config)
        >>> # Returns: {"lr": 0.0001234, "batch_size": 64, "epochs": 50}
    """
    params = {}

    for param_name, config in param_config.items():
        param_type = config["type"]

        if param_type == "float":
            log_scale = config.get("log", False)
            params[param_name] = trial.suggest_float(
                param_name, config["low"], config["high"], log=log_scale
            )
        elif param_type == "int":
            step = config.get("step", 1)
            params[param_name] = trial.suggest_int(
                param_name, config["low"], config["high"], step=step
            )
        elif param_type == "categorical":
            params[param_name] = trial.suggest_categorical(
                param_name, config["choices"]
            )
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    return params


def create_study_with_defaults(
    study_name: str,
    storage_url: str,
    direction: str = "minimize",
    sampler_name: str = "TPE",
    pruner_name: str = "Median",
    sampler_kwargs: Optional[Dict[str, Any]] = None,
    pruner_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Create an Optuna study with commonly used samplers and pruners.

    Args:
        study_name: Name of the study
        storage_url: Storage URL (e.g., "sqlite:///study.db")
        direction: Optimization direction ("minimize" or "maximize")
        sampler_name: Sampler type ("TPE", "Random", "CmaEs")
        pruner_name: Pruner type ("Median", "Hyperband", "None")
        sampler_kwargs: Additional sampler arguments (e.g., {"seed": 42})
        pruner_kwargs: Additional pruner arguments (e.g., {"n_startup_trials": 5})

    Returns:
        Optuna study object

    Examples:
        >>> study = create_study_with_defaults(
        ...     "vae_optimization",
        ...     "sqlite:///vae_study.db",
        ...     sampler_kwargs={"seed": 42}
        ... )
        >>> study.optimize(objective, n_trials=100)
    """
    import optuna

    # Set up sampler
    sampler_kwargs = sampler_kwargs or {}
    if sampler_name == "TPE":
        sampler = optuna.samplers.TPESampler(**sampler_kwargs)
    elif sampler_name == "Random":
        sampler = optuna.samplers.RandomSampler(**sampler_kwargs)
    elif sampler_name == "CmaEs":
        sampler = optuna.samplers.CmaEsSampler(**sampler_kwargs)
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")

    # Set up pruner
    pruner_kwargs = pruner_kwargs or {}
    if pruner_name == "Median":
        pruner = optuna.pruners.MedianPruner(**pruner_kwargs)
    elif pruner_name == "Hyperband":
        pruner = optuna.pruners.HyperbandPruner(**pruner_kwargs)
    elif pruner_name == "None":
        pruner = optuna.pruners.NopPruner()
    else:
        raise ValueError(f"Unknown pruner: {pruner_name}")

    return optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction=direction,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )


def save_best_config(
    study,
    base_config_path: str,
    output_path: str,
    param_mapping: Dict[str, str],
    additional_modifications: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save the best configuration found by Optuna to a file.

    Args:
        study: Completed Optuna study
        base_config_path: Path to the base configuration file
        output_path: Where to save the best configuration
        param_mapping: Mapping from Optuna parameter names to config keys
            e.g., {"beta": "model.init_args.beta", "lr": "model.init_args.lr"}
        additional_modifications: Additional modifications to apply to the config
            e.g., {"trainer.max_epochs": 300, "model.init_args.loss_function.init_args.reduction": "mean"}

    Examples:
        >>> param_mapping = {
        ...     "beta": "model.init_args.beta",
        ...     "lr": "model.init_args.lr"
        ... }
        >>> additional_mods = {"trainer.max_epochs": 300}
        >>> save_best_config(study, "base.yml", "best.yml", param_mapping, additional_mods)
    """
    if study.best_trial is None:
        print("No best trial found")
        return

    # Create modifications dictionary
    modifications = {}
    for optuna_param, config_key in param_mapping.items():
        if optuna_param in study.best_params:
            modifications[config_key] = study.best_params[optuna_param]

    # Add additional modifications
    if additional_modifications:
        modifications.update(additional_modifications)

    # Create and save the best configuration
    modify_config(base_config_path, modifications, output_path)

    print(f"Best configuration saved to: {output_path}")
    print(f"Best value: {study.best_value:.6f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")


def cleanup_temp_files(file_patterns: List[str], working_dir: str = ".") -> None:
    """
    Clean up temporary files matching the given patterns.

    Uses glob patterns to match files for deletion. Handles both files and
    directories safely.

    Args:
        file_patterns: List of glob patterns for files to delete
            e.g., ["trial_*.yml", "temp_*", "*.tmp"]
        working_dir: Directory to search in (default: current directory)

    Examples:
        >>> cleanup_temp_files(["trial_*.yml", "temp_logs_*"])
        Removed: trial_1.yml
        Removed: trial_2.yml
        Removed: temp_logs_experiment1
    """
    for pattern in file_patterns:
        files = glob.glob(os.path.join(working_dir, pattern))
        for file_path in files:
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
                elif os.path.isdir(file_path):
                    import shutil

                    shutil.rmtree(file_path)
                    print(f"Removed directory: {file_path}")
            except Exception as e:
                print(f"Failed to remove {file_path}: {e}")


def validate_config_modifications(
    base_config_path: str, modifications: Dict[str, Any]
) -> bool:
    """
    Validate that configuration modifications are applicable to the base config.

    Checks if the nested keys exist in the base configuration structure.

    Args:
        base_config_path: Path to the base configuration file
        modifications: Dictionary of modifications to validate

    Returns:
        True if all modifications are valid, False otherwise

    Examples:
        >>> modifications = {"model.init_args.beta": 10, "invalid.key": 5}
        >>> validate_config_modifications("config.yml", modifications)
        False  # because "invalid.key" doesn't exist in base config
    """
    try:
        with open(base_config_path, "r") as f:
            config = yaml.safe_load(f)

        for key_path in modifications.keys():
            keys = key_path.split(".")
            current = config

            # Check if nested path exists
            for key in keys[:-1]:
                if not isinstance(current, dict) or key not in current:
                    print(f"Invalid key path: {key_path} (missing: {key})")
                    return False
                current = current[key]

            # Check final key (it's ok if it doesn't exist, we'll create it)
            if not isinstance(current, dict):
                print(f"Invalid key path: {key_path} (parent is not dict)")
                return False

        return True

    except Exception as e:
        print(f"Error validating config modifications: {e}")
        return False
