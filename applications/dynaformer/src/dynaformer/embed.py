"""Cache a frozen foundation model's embedding of every track point, under the 8 rotations/flips of the square.

Usage
-----
python -m dynaformer.embed --data-dir <Bill/5-cellcycle> --encoder c-radio --out <Bill/5-cellcycle/embeddings>

Every row of ``crops/phase.npy`` is decoded, z-scored with the whole-mosaic statistics, centre-cropped to
``--img-size`` and embedded by :class:`~dynaformer.models.FoundationFrameEncoder` once per view
(:func:`~dynaformer.cellcycle_data.d4_view`). Writes ``<out>/<encoder>.npy`` of shape ``(N, 8, embed_dim)``
(float16, rows aligned with ``points.csv``) and ``<out>/<encoder>.json`` with the widths and settings.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dynaformer.cellcycle_data import N_VIEWS, d4_view, decode_phase
from dynaformer.models.foundation import FOUNDATION_MODELS, FoundationFrameEncoder

_logger = logging.getLogger(__name__)

#: Layout of every embedding cache.
LAYOUT = "(row, view, [global token | central patch-token mean]), float16"


def encoder_meta(encoder: FoundationFrameEncoder) -> dict:
    """Return the settings and widths of an encoder, as recorded next to its cache."""
    return {
        "encoder": encoder.name,
        "weights": encoder.weights,
        "img_size": encoder.img_size,
        "center_patches": encoder.center_patches,
        "rgb_window": list(encoder.rgb_window),
        "token_dim": encoder.token_dim,
        "patch_dim": encoder.patch_dim,
        "embed_dim": encoder.embed_dim,
        "views": N_VIEWS,
        "layout": LAYOUT,
    }


def embed_views(encoder: FoundationFrameEncoder, frames: torch.Tensor) -> torch.Tensor:
    """Embed ``(B, 1, Y, X)`` frames under each of the 8 views: ``(B, 8, embed_dim)``, float32.

    Raises
    ------
    RuntimeError
        If any embedding is not finite.
    """
    views = torch.cat([d4_view(frames, view) for view in range(N_VIEWS)])
    embeddings = encoder(views).float().view(N_VIEWS, len(frames), -1).transpose(0, 1)
    if not torch.isfinite(embeddings).all():
        raise RuntimeError("Non-finite embeddings.")
    return embeddings


class _CentredPhase(Dataset):
    """Decoded, z-scored, centre-cropped phase of every stored crop."""

    def __init__(self, path: Path, stats: dict, size: int) -> None:
        self.path = path
        self.stats = stats
        self.size = size
        self.margin = (stats["crop"] - size) // 2
        self.n = len(np.load(path, mmap_mode="r"))
        self.phase = None

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, row: int) -> torch.Tensor:
        if self.phase is None:
            self.phase = np.load(self.path, mmap_mode="r")
        window = slice(self.margin, self.margin + self.size)
        crop = np.asarray(self.phase[row, window, window])
        return torch.from_numpy(decode_phase(crop, self.stats["phase"]))[None]


def main() -> None:
    """Embed every track point and write the cache."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--encoder", choices=sorted(FOUNDATION_MODELS), required=True)
    parser.add_argument("--weights", default=None, help="override the default weights of the encoder")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64, help="track points per step; each is run 8 times")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    with open(args.data_dir / "stats.json") as f:
        stats = json.load(f)
    dataset = _CentredPhase(args.data_dir / "crops" / "phase.npy", stats, args.img_size)
    device = torch.device("cuda")
    encoder = FoundationFrameEncoder(args.encoder, weights=args.weights, img_size=args.img_size).to(device)
    meta = {**encoder_meta(encoder), "n_points": len(dataset)}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.encoder}.npy"
    header = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float16, shape=(len(dataset), N_VIEWS, encoder.embed_dim)
    )
    offset, row_bytes = header.offset, N_VIEWS * encoder.embed_dim * 2
    del header

    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    start, row = time.time(), 0
    with open(path, "r+b") as out, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for step, phase in enumerate(loader):
            embeddings = embed_views(encoder, phase.to(device, non_blocking=True))
            out.seek(offset + row * row_bytes)
            out.write(embeddings.half().cpu().numpy().tobytes())
            row += len(phase)
            if step % 200 == 0:
                rate = row / (time.time() - start)
                _logger.info(f"{row}/{len(dataset)} points, {rate * N_VIEWS:.0f} images/s")
    if row != len(dataset):
        raise RuntimeError(f"Embedded {row} of {len(dataset)} points.")
    with open(args.out / f"{args.encoder}.json", "w") as f:
        json.dump(meta, f, indent=1)
    _logger.info(f"Wrote {path} in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
