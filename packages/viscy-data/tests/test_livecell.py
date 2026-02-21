import json

import numpy as np
import pytest
import torch

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("pycocotools")
pytest.importorskip("torchvision")

from pycocotools import mask as mask_utils  # noqa: E402

from viscy_data import LiveCellDataModule, LiveCellDataset, LiveCellTestDataset  # noqa: E402

IMG_H, IMG_W = 64, 64
N_IMAGES = 4


def _make_coco_json(path, image_names, with_annotations=True):
    images = [{"id": i, "file_name": name, "height": IMG_H, "width": IMG_W} for i, name in enumerate(image_names)]
    annotations = []
    if with_annotations:
        for i, _ in enumerate(image_names):
            binary_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
            binary_mask[10:30, 10:30] = 1
            rle = mask_utils.encode(np.asfortranarray(binary_mask))
            rle["counts"] = rle["counts"].decode("utf-8")
            annotations.append(
                {
                    "id": i,
                    "image_id": i,
                    "bbox": [10, 10, 20, 20],
                    "area": 400,
                    "segmentation": rle,
                    "category_id": 1,
                    "iscrowd": 0,
                }
            )
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "cell"}],
    }
    with open(path, "w") as f:
        json.dump(coco, f)


@pytest.fixture()
def livecell_data(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    image_names = [f"img_{i:03d}.tiff" for i in range(N_IMAGES)]
    for name in image_names:
        rng = np.random.default_rng()
        img = rng.integers(0, 255, (IMG_H, IMG_W), dtype=np.uint8)
        tifffile.imwrite(img_dir / name, img)
    train_names = image_names[:2]
    val_names = image_names[2:3]
    test_names = image_names[3:]
    train_json = tmp_path / "train.json"
    val_json = tmp_path / "val.json"
    test_json = tmp_path / "test.json"
    _make_coco_json(train_json, train_names)
    _make_coco_json(val_json, val_names)
    _make_coco_json(test_json, test_names)
    return img_dir, train_json, val_json, test_json


def test_livecell_dataset_len(livecell_data):
    img_dir, train_json, _, _ = livecell_data
    with open(train_json) as f:
        image_names = [img["file_name"] for img in json.load(f)["images"]]
    cache_map = torch.multiprocessing.Manager().dict()
    ds = LiveCellDataset(
        images=[img_dir / name for name in image_names],
        transform=lambda x: x,
        cache_map=cache_map,
    )
    assert len(ds) == len(image_names)


def test_livecell_dataset_getitem(livecell_data):
    img_dir, train_json, _, _ = livecell_data
    with open(train_json) as f:
        image_names = [img["file_name"] for img in json.load(f)["images"]]
    cache_map = torch.multiprocessing.Manager().dict()
    ds = LiveCellDataset(
        images=[img_dir / name for name in image_names],
        transform=lambda x: x,
        cache_map=cache_map,
    )
    sample = ds[0]
    assert isinstance(sample, list)
    assert "source" in sample[0]
    assert sample[0]["source"].shape == (1, 1, IMG_H, IMG_W)


def test_livecell_dataset_caching(livecell_data):
    img_dir, train_json, _, _ = livecell_data
    with open(train_json) as f:
        image_names = [img["file_name"] for img in json.load(f)["images"]]
    paths = [img_dir / name for name in image_names]
    cache_map = torch.multiprocessing.Manager().dict()
    ds = LiveCellDataset(images=paths, transform=lambda x: x, cache_map=cache_map)
    assert paths[0] not in cache_map
    _ = ds[0]
    assert paths[0] in cache_map


def test_livecell_test_dataset_basic(livecell_data):
    img_dir, _, _, test_json = livecell_data
    from monai.transforms import Compose

    ds = LiveCellTestDataset(
        image_dir=img_dir,
        transform=Compose([]),
        annotations=test_json,
    )
    assert len(ds) == 1
    sample = ds[0]
    assert "source" in sample
    assert sample["source"].shape == (1, 1, IMG_H, IMG_W)


def test_livecell_test_dataset_with_labels(livecell_data):
    img_dir, _, _, test_json = livecell_data
    from monai.transforms import Compose

    ds = LiveCellTestDataset(
        image_dir=img_dir,
        transform=Compose([]),
        annotations=test_json,
        load_labels=True,
    )
    sample = ds[0]
    assert "detections" in sample
    dets = sample["detections"]
    assert "boxes" in dets
    assert "labels" in dets
    assert "masks" in dets
    assert dets["boxes"].ndim == 2
    assert dets["boxes"].shape[1] == 4
    assert dets["masks"].dtype == torch.bool


def test_livecell_test_dataset_transform_applied(livecell_data):
    img_dir, _, _, test_json = livecell_data

    class MarkTransform:
        def __call__(self, sample):
            sample["transformed"] = True
            return sample

    ds = LiveCellTestDataset(
        image_dir=img_dir,
        transform=MarkTransform(),
        annotations=test_json,
    )
    sample = ds[0]
    assert sample.get("transformed") is True, "Transform result was not assigned back"


def test_livecell_datamodule_setup_fit(livecell_data):
    img_dir, train_json, val_json, _ = livecell_data
    dm = LiveCellDataModule(
        train_val_images=img_dir,
        train_annotations=train_json,
        val_annotations=val_json,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
    )
    dm.setup("fit")
    assert hasattr(dm, "train_dataset")
    assert hasattr(dm, "val_dataset")
    assert len(dm.train_dataset) == 2
    assert len(dm.val_dataset) == 1


def test_livecell_datamodule_setup_test(livecell_data):
    img_dir, _, _, test_json = livecell_data
    dm = LiveCellDataModule(
        test_images=img_dir,
        test_annotations=test_json,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
    )
    dm.setup("test")
    assert hasattr(dm, "test_dataset")
    assert len(dm.test_dataset) == 1
