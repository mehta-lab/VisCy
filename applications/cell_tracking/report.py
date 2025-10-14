#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "polars",
#   "altair>=5.4.0",
# ]
# ///
import polars as pl
import altair as alt


def main() -> None:
    df = pl.read_csv("results.csv")
    gdf = df.group_by("model", "dataset").mean()

    alt.renderers.enable("browser")  # use system browser

    metric = "OP_CLB(0)"
    p = gdf.plot.bar(x="model", y=metric, color="model")
    p = p.properties(
        title=metric,
    )

    facet_chart = p.facet(
        facet="dataset",
        columns=5,
    )

    facet_chart.show()

    df.group_by("model").mean().plot.bar(x="model", y=metric, color="model").properties(
        title=metric,
    ).show()


if __name__ == "__main__":
    main()