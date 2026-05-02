// Gather per-experiment smoothness CSVs into one combined file + per-marker summary.
// Runs after all SMOOTHNESS scatter jobs finish.

process SMOOTHNESS_GATHER {
    executor 'local'

    input:
    val smoothness_dir
    val done  // signal that all SMOOTHNESS jobs finished

    output:
    val smoothness_dir, emit: smoothness_dir

    script:
    """
    uv run --project=${params.workspace_dir} --package=dynaclr python3 -c "
import glob, os
import pandas as pd

smoothness_dir = '${smoothness_dir}'
csvs = glob.glob(os.path.join(smoothness_dir, '*_per_marker_smoothness.csv'))
if not csvs:
    raise RuntimeError(f'No per_marker_smoothness CSVs found in {smoothness_dir}')

combined = pd.concat([pd.read_csv(f) for f in sorted(csvs)], ignore_index=True)
out = os.path.join(smoothness_dir, 'all_experiments_per_marker_smoothness.csv')
combined.to_csv(out, index=False)
print(f'Wrote {len(combined)} rows from {len(csvs)} experiments to {out}')

# Per-marker summary: mean +/- std across experiments
metric_cols = [c for c in combined.columns if c not in ('experiment', 'marker')]
agg = combined.groupby('marker')[metric_cols].agg(['mean', 'std'])
agg.columns = ['_'.join(c) for c in agg.columns]
agg = agg.reset_index()
summary_out = os.path.join(smoothness_dir, 'per_marker_summary.csv')
agg.to_csv(summary_out, index=False)
print(f'Per-marker summary ({len(agg)} markers) written to {summary_out}')
"
    """
}
