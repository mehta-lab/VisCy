# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from viscy.representation.embedding_writer import read_embedding_dataset

# %%
train_annotations = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_11_26_A549_ZIKA-sensor_ZIKV/3-phenotype/annotate-infection/combined_annotations.csv"
)
train_embeddings = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/bootstrap-labels/generate-labels/sensor-2024-11-26.zarr"
)
val_annotations = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/2024_08_14_ZIKV_pal17_48h/6-phenotype/combined_annotations.csv"
    # "/hpc/projects/intracellular_dashboard/viral-sensor/2024_11_05_A549_pAL10_24h/4-phenotype/annotate-infection/combined_annotations.csv"
)
val_embeddings = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/bootstrap-labels/generate-labels/sensor-2024-08-14-annotation.zarr"
    # "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/bootstrap-labels/generate-labels/sensor-2024-11-05.zarr"
)


# %%
def filter_train_fovs(fov_name: pd.Series) -> pd.Series:
    return fov_name.str[1:4].isin(["C/2", "B/3"])


def filter_val_fovs(fov_name: pd.Series) -> pd.Series:
    return fov_name.str[1:4].isin(["0/3"]) | (fov_name == "/0/4/000001")
    # return fov_name.isin(
    #     ["/0/15/000001", "/0/11/002000", "/0/11/002001", "/0/11/002002"]
    # )


def all_fovs(fov_name: pd.Series) -> pd.Series:
    return None


def load_features_and_annotations(embedding_path, annotation_path, filter_fn):
    dataset = read_embedding_dataset(embedding_path)
    features = dataset["features"][filter_fn(dataset["fov_name"])]
    annotation = pd.read_csv(annotation_path)
    annotation["fov_name"] = "/" + annotation["fov_name"]
    annotation = annotation.set_index(["fov_name", "id"])
    index = features["sample"].to_dataframe().reset_index(drop=True)[["fov_name", "id"]]
    selected = pd.merge(
        left=index, right=annotation, on=["fov_name", "id"], how="inner"
    )
    selected["infection_state"] = selected["infection_state"].astype("category")
    return features, selected["infection_state"], selected


# %%
train_features, train_annotation, train_selected = load_features_and_annotations(
    train_embeddings, train_annotations, filter_fn=filter_train_fovs
)
val_features, val_annotation, val_selected = load_features_and_annotations(
    val_embeddings, val_annotations, filter_fn=filter_val_fovs
)

model = LogisticRegression(class_weight="balanced", random_state=42, solver="liblinear")
model = model.fit(train_features, train_annotation)
train_prediction = model.predict(train_features)
val_prediction = model.predict(val_features)

print("Training\n", classification_report(train_annotation, train_prediction))
print("Validation\n", classification_report(val_annotation, val_prediction))

val_selected["label"] = val_selected["infection_state"].cat.codes
val_selected["prediction_binary"] = val_prediction

# %%
prediction = val_selected
prediction["HPI"] = prediction["t"] / 2 + 3
bins = [3, 6, 9, 12, 15, 18, 21, 24]
labels = [f"{start}-{end}" for start, end in zip(bins[:-1], bins[1:])]
prediction["stage"] = pd.cut(prediction["HPI"], bins=bins, labels=labels, right=True)
comparison = prediction.melt(
    id_vars=["fov_name", "id", "HPI"],
    value_vars=["label", "prediction_binary"],
    var_name="source",
    value_name="value",
)
with sns.axes_style("whitegrid"):
    ax = sns.lineplot(
        data=comparison,
        x="HPI",
        y="value",
        style="source",
        errorbar=None,
        color="gray",
    )
    ax.set_ylabel("Infection ratio")

# %%
accuracy_by_t = prediction.groupby(["stage"]).apply(
    lambda x: float(accuracy_score(x["label"], x["prediction_binary"]))
)
f1_by_t = prediction.groupby(["stage"]).apply(
    lambda x: float(f1_score(x["label"], x["prediction_binary"]))
)

metrics_df = pd.DataFrame(
    data={
        "accuracy": accuracy_by_t.values,
        "F1": f1_by_t.values,
    },
    index=f1_by_t.index,
).reset_index()

metrics_long = metrics_df.melt(
    id_vars=["stage"],
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
        kind="point",
        linewidth=1.5,
        linestyles="--",
        legend=False,
        color="gray",
    )
    g.set_axis_labels("HPI", "accuracy")
    g.figure.set_size_inches(3.5, 0.75)
    g.set(xlim=(-1, 7), ylim=(0.9, 1.0))
    plt.show()

# %%
g.savefig(
    Path.home()
    / "gdrive/publications/dynaCLR/2025_dynaCLR_paper/fig_manuscript_svg/figure_knowledge_distillation/figure_parts/teacher_accuracy.pdf",
    dpi=300,
    bbox_inches="tight",
)

# %%
