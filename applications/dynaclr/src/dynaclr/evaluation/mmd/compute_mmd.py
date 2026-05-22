"""CLI for computing MMD² between two pooled groups of cell embeddings.

Usage
-----
    dynaclr compute-mmd -c config.yaml

Each group is defined by one or more wells. A well is identified by its
zarr_path (AnnData store) plus well_name and well_id, which are matched
against the ``fov_name`` obs column (format: ``well_name/well_id/pos_id``).
Embeddings from all wells in a group are pooled before MMD² is computed.

Example config (YAML)
---------------------
    group_a:
      - zarr_path: /path/to/experiment1_embeddings.zarr
        well_name: B03
        well_id: 1
      - zarr_path: /path/to/experiment2_embeddings.zarr
        well_name: B03
        well_id: 1

    group_b:
      - zarr_path: /path/to/experiment3_embeddings.zarr
        well_name: C04
        well_id: 2

    n_perm: 1000
    max_cells: 2000
    random_seed: 42
    output_csv: /path/to/mmd_results.csv
"""

import logging
from pathlib import Path

import click
from pydantic import ValidationError

from dynaclr.evaluation.mmd.config import MMDConfig
from dynaclr.evaluation.mmd.mmd import compute_mmd
from viscy_utils.cli_utils import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to YAML configuration file.",
)
def main(config: Path):
    """Compute MMD² between two pooled groups of cell embeddings."""
    click.echo("=" * 60)
    click.echo("MMD² COMPUTATION")
    click.echo("=" * 60)

    try:
        config_dict = load_config(config)
        cfg = MMDConfig(**config_dict)
    except ValidationError as e:
        click.echo(f"\nConfiguration validation failed:\n{e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"\nFailed to load configuration: {e}", err=True)
        raise click.Abort()

    click.echo(f"\nGroup A: {len(cfg.group_a)} well(s)")
    for w in cfg.group_a:
        click.echo(f"  {w.zarr_path}  —  {w.well_name}/{w.well_id}")
    click.echo(f"Group B: {len(cfg.group_b)} well(s)")
    for w in cfg.group_b:
        click.echo(f"  {w.zarr_path}  —  {w.well_name}/{w.well_id}")
    click.echo(
        f"\nn_perm={cfg.n_perm}  max_cells={cfg.max_cells}  seed={cfg.random_seed}"
    )

    try:
        result = compute_mmd(
            group_a=cfg.group_a_as_dicts(),
            group_b=cfg.group_b_as_dicts(),
            n_perm=cfg.n_perm,
            max_cells=cfg.max_cells,
            random_seed=cfg.random_seed,
        )
    except Exception as e:
        click.echo(f"\nMMD computation failed: {e}", err=True)
        raise click.Abort()

    p_str = f"{result['p_value']:.4f}" if result["p_value"] is not None else "n/a"
    click.echo("\n── Results ──────────────────────────────────────")
    click.echo(f"  n_a    : {result['n_a']}")
    click.echo(f"  n_b    : {result['n_b']}")
    click.echo(f"  MMD²   : {result['mmd2']:.6f}")
    click.echo(f"  p-value: {p_str}")
    click.echo(f"  gamma  : {result['gamma']:.6f}")

    if cfg.output_csv:
        import csv

        output = Path(cfg.output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["n_a", "n_b", "mmd2", "p_value", "gamma"]
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(
                {
                    "n_a": result["n_a"],
                    "n_b": result["n_b"],
                    "mmd2": result["mmd2"],
                    "p_value": result["p_value"] if result["p_value"] is not None else "",
                    "gamma": result["gamma"],
                }
            )
        click.echo(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
