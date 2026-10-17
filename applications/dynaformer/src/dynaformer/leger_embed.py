"""Cache a frozen foundation model's embedding of every Leger et al. frame, in chunks that fit short GPU jobs.

Usage
-----
python -m dynaformer.leger_embed index --data-dir <data> --out <data>/embeddings
python -m dynaformer.leger_embed plan --out <data>/embeddings --encoder c-radio --frames-per-chunk 80000
python -m dynaformer.leger_embed chunk --data-dir <data> --out <data>/embeddings --encoder c-radio --chunk 3
python -m dynaformer.leger_embed finalize --out <data>/embeddings --encoder c-radio

``index`` writes ``index.csv`` and ``labels.npy`` (:func:`~dynaformer.leger_data.write_index`). ``plan`` builds the
encoder once, creates the cache ``<encoder>.npy`` ``(N, 8, embed_dim)`` float16 (rows aligned with the index) and
writes ``<encoder>.plan.json``: the encoder's settings, a fingerprint of its weights, and row ranges of whole tracks.
Randomly drawn weights (``vit-untrained``) are saved with the plan: the same seed drew different weights on the
Intel GPU nodes than on the AMD login node. Each ``chunk`` job embeds one range, every frame under the 8 views of
:func:`~dynaformer.cellcycle_data.d4_view` as in :mod:`dynaformer.embed`, checks that it built the planned weights,
and leaves ``<encoder>.chunks/<k>.json`` when its rows are written. ``finalize`` checks every chunk's marker and every
row of the cache (finite, not left empty), then writes ``<encoder>.json``;
:class:`~dynaformer.leger_data.LegerDataModule` reads nothing without it.

The input is the stored channel (brightfield by default), per-frame z-scored as published, at its 64 px; the encoder
resizes it to ``img_size``. ``rgb_window`` maps z-scores to the RGB models' ``[0, 1]`` range: the phase default
(-4, 8) leaves brightfield in the lower two thirds, so ``plan`` takes the window as an argument.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dynaformer.cellcycle_data import N_VIEWS
from dynaformer.embed import embed_views, encoder_meta
from dynaformer.leger_data import LegerTrackFrames, index_fingerprint, load_index, plan_chunks, track_table, write_index
from dynaformer.models.foundation import FOUNDATION_MODELS, FoundationFrameEncoder

_logger = logging.getLogger(__name__)

#: Encoders whose weights are drawn at random rather than loaded: the plan saves them for every chunk to load.
DRAWN_WEIGHTS = ("vit-untrained",)


def weights_fingerprint(module: torch.nn.Module) -> list[float]:
    """Sum and absolute sum of every parameter and buffer, in float64: equal weights give equal fingerprints."""
    total = absolute = 0.0
    for tensor in (*module.parameters(), *module.buffers()):
        tensor = tensor.detach().double()
        total += float(tensor.sum())
        absolute += float(tensor.abs().sum())
    return [total, absolute]


def build_encoder(settings: dict) -> FoundationFrameEncoder:
    """Build the planned encoder on the CPU."""
    return FoundationFrameEncoder(
        settings["encoder"],
        weights=settings["weights"],
        img_size=settings["img_size"],
        center_patches=settings["center_patches"],
        rgb_window=tuple(settings["rgb_window"]),
    )


def plan(
    out: Path,
    encoder_name: str,
    frames_per_chunk: int,
    channel: int = 0,
    weights: str | None = None,
    img_size: int = 224,
    rgb_window: tuple[float, float] = (-4.0, 8.0),
    encoder: FoundationFrameEncoder | None = None,
) -> dict:
    """Create the cache and its chunk plan.

    Parameters
    ----------
    out : Path
        Folder holding ``index.csv``; the cache is written there.
    encoder_name : str
        Name of the foundation model.
    frames_per_chunk : int
        Most frames per chunk; chunks hold whole tracks.
    channel : int
        Image channel embedded, by default 0 (brightfield).
    weights, img_size, rgb_window
        Encoder settings (:class:`~dynaformer.models.FoundationFrameEncoder`).
    encoder : FoundationFrameEncoder | None
        An already-built encoder to plan for instead of building one from the settings, by default ``None``.

    Returns
    -------
    dict
        The plan, also written to ``<encoder>.plan.json``.

    Raises
    ------
    FileExistsError
        If a cache or plan of this encoder already exists; remove it to start over.
    """
    path = out / f"{encoder_name}.npy"
    weights_file = path.with_suffix(".weights.pt")
    for existing in (
        path,
        path.with_suffix(".plan.json"),
        path.with_suffix(".json"),
        weights_file,
        path.parent / f"{encoder_name}.chunks",
    ):
        if existing.exists():
            raise FileExistsError(f"{existing} exists; remove the {encoder_name} cache, plan and chunks to replan.")
    index, _ = load_index(out)
    if encoder is None:
        encoder = FoundationFrameEncoder(encoder_name, weights=weights, img_size=img_size, rgb_window=rgb_window)
    settings = encoder_meta(encoder)
    chunks = plan_chunks(track_table(index), frames_per_chunk)
    result = {
        **settings,
        "channel": channel,
        "n_points": len(index),
        "index_sha1": index_fingerprint(out),
        "weights_fingerprint": weights_fingerprint(encoder),
        "weights_file": weights_file.name if encoder_name in DRAWN_WEIGHTS else None,
        "chunks": chunks,
    }
    if result["weights_file"] is not None:
        torch.save(encoder.state_dict(), weights_file)
    np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(index), N_VIEWS, encoder.embed_dim))
    path.with_suffix(".plan.json").write_text(json.dumps(result, indent=1))
    (out / f"{encoder_name}.chunks").mkdir()
    return result


def embed_chunk(
    data_dir: Path,
    out: Path,
    encoder_name: str,
    chunk: int,
    batch_size: int = 64,
    num_workers: int = 8,
    device: str = "cuda",
    encoder: FoundationFrameEncoder | None = None,
) -> dict:
    """Embed one planned chunk into the cache and leave its marker.

    Parameters
    ----------
    data_dir : Path
        Dataset root.
    out : Path
        Folder of the cache and its plan.
    encoder_name : str
        Name of the planned foundation model.
    chunk : int
        Chunk number in the plan.
    batch_size : int
        Frames per forward pass; each is run under 8 views, by default 64.
    num_workers : int
        Track-reading worker processes, by default 8.
    device : str
        Torch device, by default ``"cuda"``.
    encoder : FoundationFrameEncoder | None
        An already-built encoder to use instead of building the planned one, by default ``None``. Drawn weights saved
        with the plan are loaded into it, and its weights must then match the plan's fingerprint.

    Returns
    -------
    dict
        The chunk's marker: its rows, duration and throughput.
    """
    path = out / f"{encoder_name}.npy"
    planned = json.loads(path.with_suffix(".plan.json").read_text())
    if planned["index_sha1"] != index_fingerprint(out):
        raise ValueError(f"{out / 'index.csv'} changed since {encoder_name} was planned.")
    start, end = planned["chunks"][chunk]
    if encoder is None:
        encoder = build_encoder(planned)
    if planned["weights_file"] is not None:
        encoder.load_state_dict(torch.load(out / planned["weights_file"], weights_only=True))
    if not np.allclose(weights_fingerprint(encoder), planned["weights_fingerprint"], rtol=1e-12, atol=0):
        raise ValueError(f"The {encoder_name} weights built here differ from the planned ones.")
    encoder = encoder.to(device)
    index, _ = load_index(out)
    tracks = track_table(index)
    tracks = tracks[(tracks.row >= start) & (tracks.row < end)]
    dataset = LegerTrackFrames(data_dir, tracks, channel=planned["channel"])
    loader = DataLoader(dataset, batch_size=None, num_workers=num_workers, pin_memory=device != "cpu")

    cache = np.load(path, mmap_mode="r")
    if cache.shape != (planned["n_points"], N_VIEWS, planned["embed_dim"]):
        raise ValueError(f"{path} has shape {cache.shape}, not the planned one.")
    offset, row_bytes = cache.offset, N_VIEWS * planned["embed_dim"] * 2
    del cache

    began = time.time()
    pending: list[torch.Tensor] = []  # frames read but not yet embedded, starting at row `written`
    written = start
    autocast = torch.autocast(device, dtype=torch.bfloat16, enabled=device != "cpu")
    with open(path, "r+b") as handle, torch.inference_mode(), autocast:

        def flush(n: int) -> None:
            nonlocal written, pending
            frames = torch.cat(pending)
            pending = [frames[n:]] if n < len(frames) else []
            try:
                embeddings = embed_views(encoder, frames[:n].to(device, non_blocking=True))
            except RuntimeError as error:
                raise RuntimeError(f"Rows {written}-{written + n} of {encoder_name}: {error}") from error
            handle.seek(offset + written * row_bytes)
            handle.write(embeddings.half().cpu().numpy().tobytes())
            written += n

        buffered = 0
        for row, frames in loader:
            if row != written + buffered:
                raise RuntimeError(f"Expected the track at row {written + buffered}, read the one at {row}.")
            pending.append(frames)
            buffered += len(frames)
            while buffered >= batch_size:
                flush(batch_size)
                buffered -= batch_size
            if (written - start) % (100 * batch_size) < batch_size:
                rate = (written - start) * N_VIEWS / max(time.time() - began, 1e-9)
                _logger.info(f"rows {written}/{end} of chunk {chunk} [{start}, {end}), {rate:.0f} images/s")
        if buffered:
            flush(buffered)
    if written != end:
        raise RuntimeError(f"Chunk {chunk} wrote rows up to {written}, planned up to {end}.")
    seconds = time.time() - began
    marker = {
        "chunk": chunk,
        "rows": [start, end],
        "seconds": round(seconds, 1),
        "images_per_s": round((end - start) * N_VIEWS / seconds, 1),
        "device": torch.cuda.get_device_name() if device != "cpu" else "cpu",
    }
    (out / f"{encoder_name}.chunks" / f"{chunk}.json").write_text(json.dumps(marker, indent=1))
    return marker


def finalize(out: Path, encoder_name: str, block: int = 16384) -> dict:
    """Check every chunk and every row of the cache, then write ``<encoder>.json``.

    Raises
    ------
    RuntimeError
        If a chunk has no marker, or a row is not finite or was never written (all zeros).
    """
    path = out / f"{encoder_name}.npy"
    planned = json.loads(path.with_suffix(".plan.json").read_text())
    markers = []
    for chunk, rows in enumerate(planned["chunks"]):
        marker_path = out / f"{encoder_name}.chunks" / f"{chunk}.json"
        if not marker_path.exists():
            raise RuntimeError(f"Chunk {chunk} {rows} of {encoder_name} has not been written.")
        marker = json.loads(marker_path.read_text())
        if marker["rows"] != list(rows):
            raise RuntimeError(f"Chunk {chunk} of {encoder_name} wrote rows {marker['rows']}, planned {rows}.")
        markers.append(marker)
    cache = np.load(path, mmap_mode="r")
    for first in range(0, len(cache), block):
        values = np.asarray(cache[first : first + block])
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite values in {path} rows {first}-{first + len(values)}.")
        empty = ~values.any(axis=(1, 2))
        if empty.any():
            raise RuntimeError(f"Row {first + int(np.argmax(empty))} of {path} was never written.")
    meta = {key: value for key, value in planned.items() if key != "chunks"}
    meta["chunks"] = markers
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return meta


def main() -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="write index.csv and labels.npy")
    index.add_argument("--data-dir", type=Path, required=True)
    index.add_argument("--out", type=Path, required=True)
    planner = commands.add_parser("plan", help="create the cache and its chunk plan")
    planner.add_argument("--out", type=Path, required=True)
    planner.add_argument("--encoder", choices=sorted(FOUNDATION_MODELS), required=True)
    planner.add_argument("--frames-per-chunk", type=int, required=True)
    planner.add_argument("--channel", type=int, default=0, help="0 brightfield, 1 H2B")
    planner.add_argument("--weights", default=None, help="override the default weights of the encoder")
    planner.add_argument("--img-size", type=int, default=224)
    planner.add_argument("--rgb-window", type=float, nargs=2, default=(-4.0, 8.0))
    chunker = commands.add_parser("chunk", help="embed one planned chunk")
    chunker.add_argument("--data-dir", type=Path, required=True)
    chunker.add_argument("--out", type=Path, required=True)
    chunker.add_argument("--encoder", choices=sorted(FOUNDATION_MODELS), required=True)
    chunker.add_argument("--chunk", type=int, required=True)
    chunker.add_argument("--batch-size", type=int, default=64, help="frames per step; each is run 8 times")
    chunker.add_argument("--num-workers", type=int, default=8)
    finisher = commands.add_parser("finalize", help="check every chunk and row, then write <encoder>.json")
    finisher.add_argument("--out", type=Path, required=True)
    finisher.add_argument("--encoder", choices=sorted(FOUNDATION_MODELS), required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.command == "index":
        written = write_index(args.data_dir, args.out)
        _logger.info(f"Indexed {len(written)} frames of {written.track_index.nunique()} tracks in {args.out}")
    elif args.command == "plan":
        result = plan(
            args.out,
            args.encoder,
            args.frames_per_chunk,
            channel=args.channel,
            weights=args.weights,
            img_size=args.img_size,
            rgb_window=tuple(args.rgb_window),
        )
        _logger.info(f"Planned {len(result['chunks'])} chunks of {args.encoder} ({result['embed_dim']} wide)")
        print(len(result["chunks"]))
    elif args.command == "chunk":
        marker = embed_chunk(
            args.data_dir, args.out, args.encoder, args.chunk, batch_size=args.batch_size, num_workers=args.num_workers
        )
        _logger.info(f"Chunk {args.chunk} of {args.encoder}: {marker}")
    else:
        meta = finalize(args.out, args.encoder)
        _logger.info(f"{args.encoder}: all {len(meta['chunks'])} chunks and {meta['n_points']} rows check out")


if __name__ == "__main__":
    main()
