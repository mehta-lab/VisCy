#!/usr/bin/env python3
"""
Optuna hyperparameter optimization for VAE training with PyTorch Lightning.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

import click
import optuna
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_best_val_loss(log_dir: str) -> float:
    """Extract the best validation loss from TensorBoard logs."""
    try:
        # Find the events file
        events_files = list(Path(log_dir).glob("events.out.tfevents.*"))
        if not events_files:
            print(f"Warning: No events file found in {log_dir}")
            return float("inf")

        # Load TensorBoard data
        ea = EventAccumulator(str(events_files[0]))
        ea.Reload()

        # Extract validation loss
        if "loss/total/val" in ea.Tags()["scalars"]:
            val_losses = ea.Scalars("loss/total/val")
            best_val_loss = min([scalar.value for scalar in val_losses])
            print(f"Best validation loss: {best_val_loss:.6f}")
            return best_val_loss
        else:
            print(f"Warning: No validation loss found in {log_dir}")
            return float("inf")

    except Exception as e:
        print(f"Error extracting validation loss from {log_dir}: {e}")
        return float("inf")


def create_trial_config(
    base_config_path: str, trial: optuna.Trial, trial_dir: Path
) -> str:
    """Create a modified config file for the current trial."""

    # Load base config
    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)

    # Sample hyperparameters
    beta = trial.suggest_float("beta", 0.1, 50.0, log=True)
    lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
    warmup_epochs = trial.suggest_int("warmup_epochs", 10, 100)
    latent_dim = trial.suggest_categorical("latent_dim", [512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # Modify model config
    config["model"]["init_args"]["beta"] = beta
    config["model"]["init_args"]["lr"] = lr
    config["model"]["init_args"]["beta_warmup_epochs"] = warmup_epochs

    # Modify data config
    config["data"]["init_args"]["batch_size"] = batch_size

    # Reduce training for faster search
    config["trainer"]["max_epochs"] = 30
    config["trainer"]["check_val_every_n_epoch"] = 2

    # Set unique logging directory
    config["trainer"]["logger"]["init_args"]["save_dir"] = str(trial_dir)
    config["trainer"]["logger"]["init_args"]["version"] = f"trial_{trial.number}"

    # Fix loss function to use mean reduction
    config["model"]["init_args"]["loss_function"]["init_args"]["reduction"] = "mean"

    # Save trial config
    trial_config_path = trial_dir / f"trial_{trial.number}_config.yml"
    with open(trial_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(
        f"Trial {trial.number} params: beta={beta:.4f}, lr={lr:.2e}, "
        f"warmup={warmup_epochs}, latent={latent_dim}, batch={batch_size}"
    )

    return str(trial_config_path)


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function."""

    # Create temporary directory for this trial
    with tempfile.TemporaryDirectory(
        prefix=f"optuna_trial_{trial.number}_"
    ) as temp_dir:
        trial_dir = Path(temp_dir)

        try:
            # Create trial config
            base_config = "/hpc/projects/organelle_phenotyping/models/SEC61B/vae/fit_phase_only.yml"
            trial_config_path = create_trial_config(base_config, trial, trial_dir)

            # Run training
            cmd = [
                "python",
                "-m",
                "viscy.cli.train",
                "fit",
                "--config",
                trial_config_path,
            ]

            print(f"Running trial {trial.number}: {' '.join(cmd)}")

            # Run with timeout to prevent hanging
            result = subprocess.run(
                cmd,
                cwd="/hpc/mydata/eduardo.hirata/repos/viscy",
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            if result.returncode != 0:
                print(
                    f"Trial {trial.number} failed with return code {result.returncode}"
                )
                print(f"STDERR: {result.stderr}")
                return float("inf")

            # Extract validation loss
            log_dir = trial_dir / f"trial_{trial.number}"
            val_loss = extract_best_val_loss(str(log_dir))

            print(f"Trial {trial.number} completed with val_loss: {val_loss:.6f}")
            return val_loss

        except subprocess.TimeoutExpired:
            print(f"Trial {trial.number} timed out")
            return float("inf")
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            return float("inf")


def main():
    """Main optimization loop."""

    # Set up study
    study_name = "vae_hyperparameter_optimization"
    storage_url = f"sqlite:///optuna_vae_study.db"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="minimize",
        load_if_exists=True,  # Resume if study exists
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    print(
        f"Starting Optuna optimization with {torch.cuda.device_count()} GPUs available"
    )
    print(f"Study storage: {storage_url}")

    try:
        # Run optimization
        study.optimize(objective, n_trials=50, timeout=24 * 3600)  # 24 hour timeout

        # Print results
        print("\nOptimization completed!")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value: {study.best_value:.6f}")
        print("Best params:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        # Save best config
        best_config_path = "best_vae_config.yml"
        base_config = (
            "/hpc/projects/organelle_phenotyping/models/SEC61B/vae/fit_phase_only.yml"
        )

        with open(base_config, "r") as f:
            config = yaml.safe_load(f)

        # Apply best parameters
        best_params = study.best_params
        config["model"]["init_args"]["beta"] = best_params["beta"]
        config["model"]["init_args"]["lr"] = best_params["lr"]
        config["model"]["init_args"]["beta_warmup_epochs"] = best_params[
            "warmup_epochs"
        ]
        config["model"]["init_args"]["encoder"]["init_args"]["latent_dim"] = (
            best_params["latent_dim"]
        )
        config["model"]["init_args"]["decoder"]["init_args"]["latent_dim"] = (
            best_params["latent_dim"]
        )
        config["data"]["init_args"]["batch_size"] = best_params["batch_size"]
        config["model"]["init_args"]["loss_function"]["init_args"]["reduction"] = "mean"

        # Restore full training settings
        config["trainer"]["max_epochs"] = 300
        config["trainer"]["check_val_every_n_epoch"] = 1

        with open(best_config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"Best configuration saved to: {best_config_path}")

    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
        print(
            f"Current best trial: {study.best_trial.number if study.best_trial else 'None'}"
        )
        if study.best_trial:
            print(f"Current best value: {study.best_value:.6f}")


if __name__ == "__main__":
    main()
