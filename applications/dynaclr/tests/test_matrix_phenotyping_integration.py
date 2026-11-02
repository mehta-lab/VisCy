"""Shared phenotyping must precede every model's downstream work."""

from dynaclr.evaluation.orchestration import matrix


def test_predict_waits_for_phenotyping_and_training(monkeypatch):
    calls = []

    def emit(stage, label, cmd, dependency, print_only):
        calls.append((stage, dependency))
        return {"train": "20", "predict": "21", "normalize": "22"}.get(stage, "23")

    monkeypatch.setattr(matrix, "_emit", emit)
    monkeypatch.setattr(matrix, "build_train_cmd", lambda model: ["train"])
    monkeypatch.setattr(matrix, "build_predict_cmd", lambda *args: ["predict"])
    monkeypatch.setattr(matrix, "build_normalize_cmd", lambda *args: ["normalize"])
    monkeypatch.setattr(matrix, "build_eval_cmd", lambda *args: ["eval"])
    model = {"family": "test", "run": "test", "ckpt_name": "last", "_phenotyping_dependencies": ["10", "11"]}
    matrix.run_model(model, ("phenotype", "train", "predict", "normalize", "eval"), False)
    assert calls == [("train", None), ("predict", "10:11:20"), ("normalize", "21"), ("eval", "22")]


def test_existing_embeddings_normalization_still_waits(monkeypatch):
    dependencies = []
    monkeypatch.setattr(matrix, "build_normalize_cmd", lambda *args: ["normalize"])
    monkeypatch.setattr(matrix, "_emit", lambda stage, label, cmd, dep, dry: dependencies.append(dep))
    model = {
        "family": "test",
        "run": "test",
        "ckpt_name": "last",
        "_skip_predict": True,
        "_phenotyping_dependencies": ["10"],
    }
    matrix.run_model(model, ("phenotype", "predict", "normalize"), False)
    assert dependencies == ["10"]
