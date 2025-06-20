# metrics for the knowledge distillation figure

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, f1_score

# %%
# Mantis
test_virus = ["C/2/000000", "C/2/001001"]
test_mock = ["B/3/000000", "B/3/000001"]

# Mantis
TRAIN_FOVS = ["C/2/000001", "C/2/001000", "B/3/001000", "B/3/001001"]

VAL_FOVS = test_virus + test_mock

# %%
prediction_from_scratch = pd.read_csv(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/bootstrap-labels/test/from-scratch-last-1126.csv"
)
prediction_from_scratch["pretraining"] = "ImageNet"

prediction_finetuned = pd.read_csv(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/bootstrap-labels/test/fine-tune-last-1126.csv"
)
pretrained_name = "DynaCLR"
prediction_finetuned["pretraining"] = pretrained_name

prediction = pd.concat([prediction_from_scratch, prediction_finetuned], axis=0)

# %%
prediction = prediction[prediction["fov_name"].isin(VAL_FOVS)]
prediction["prediction_binary"] = prediction["prediction"] > 0.5
prediction

# %%
print(
    classification_report(
        prediction["label"], prediction["prediction_binary"], digits=3
    )
)

# %%
prediction["HPI"] = prediction["t"] / 6 + 3

bins = [3, 6, 9, 12, 15, 18, 21, 24]
labels = [f"{start}-{end}" for start, end in zip(bins[:-1], bins[1:])]
prediction["stage"] = pd.cut(prediction["HPI"], bins=bins, labels=labels, right=True)
prediction["well"] = prediction["fov_name"].apply(
    lambda x: "ZIKV" if x in test_virus else "Mock"
)
comparison = prediction.melt(
    id_vars=["fov_name", "id", "HPI", "well", "stage", "pretraining"],
    value_vars=["label", "prediction_binary"],
    var_name="source",
    value_name="value",
)
with sns.axes_style("whitegrid"):
    ax = sns.lineplot(
        data=comparison[comparison["pretraining"] == pretrained_name],
        x="HPI",
        y="value",
        hue="well",
        hue_order=["Mock", "ZIKV"],
        style="source",
        errorbar=None,
        color="gray",
    )
    ax.set_ylabel("Infection ratio")

# %%
id_vars = ["stage", "pretraining"]

accuracy_by_t = prediction.groupby(id_vars).apply(
    lambda x: float(accuracy_score(x["label"], x["prediction_binary"]))
)
f1_by_t = prediction.groupby(id_vars).apply(
    lambda x: float(f1_score(x["label"], x["prediction_binary"]))
)

metrics_df = pd.DataFrame(
    data={"accuracy": accuracy_by_t.values, "F1": f1_by_t.values},
    index=f1_by_t.index,
).reset_index()

metrics_long = metrics_df.melt(
    id_vars=id_vars,
    value_vars=["accuracy"],
    var_name="metric",
    value_name="score",
)

with sns.axes_style("ticks"):
    plt.style.use("../figures/figure.mplstyle")
    g = sns.catplot(
        data=metrics_long,
        x="stage",
        y="score",
        hue="pretraining",
        kind="point",
        linewidth=1.5,
        linestyles="--",
    )
    g.set_axis_labels("HPI", "accuracy")
    sns.move_legend(g, "upper left", bbox_to_anchor=(0.35, 1.1))
    g.figure.set_size_inches(3.5, 1.5)
    g.set(xlim=(-1, 7), ylim=(0.6, 1.0))
    plt.show()


# %%
g.figure.savefig(
    Path.home()
    / "gdrive/publications/dynaCLR/2025_dynaCLR_paper/fig_manuscript_svg/figure_knowledge_distillation/figure_parts/accuracy_students.pdf",
    dpi=300,
)

# %%
