"""Optimized audio conversion utilities with parallel conversion support.

Features
- Fast ffmpeg flags for speed
- Uses ffprobe to check sample rate/channels to avoid unnecessary re-encoding
- Copies WAV files when possible (no re-encode)
- Batch/parallel conversion with ThreadPoolExecutor
- Small CLI for batch processing
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None

logger = logging.getLogger("convert_to_audio")
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_EXTENSIONS = {"mp3", "m4a", "wav", "flac", "aac", "ogg", "webm", "mp4"}


def _ffprobe_audio_info(path: Path) -> Optional[Tuple[int, int, str]]:
    """Return (sample_rate, channels, codec_name) or None on failure."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channels,sample_rate,codec_name",
        "-of",
        "json",
        str(path),
    ]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        info = json.loads(p.stdout.decode("utf-8"))
        streams = info.get("streams") or []
        if not streams:
            return None
        s = streams[0]
        sr = int(s.get("sample_rate", 0)) if s.get("sample_rate") else 0
        ch = int(s.get("channels", 0)) if s.get("channels") else 0
        codec = s.get("codec_name", "")
        return sr, ch, codec
    except Exception:
        return None


def _is_wav_copy_ok(path: Path, sample_rate: int) -> bool:
    if path.suffix.lower() != ".wav":
        return False
    info = _ffprobe_audio_info(path)
    if not info:
        return False
    sr, ch, codec = info
    # We want mono, matching sample rate and 16-bit PCM
    return sr == sample_rate and ch == 1 and codec == "pcm_s16le"


def convert_file(
    input_path: Path,
    output_path: Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    codec: str = "pcm_s16le",
    overwrite: bool = False,
) -> Tuple[Path, bool, Optional[str]]:
    """Convert a single file to WAV with desired sample rate & mono.

    Returns (output_path, success, error_message)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        logger.debug("Skipping existing output: %s", output_path)
        return output_path, True, None

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # If we can avoid re-encoding, just copy
    try:
        if _is_wav_copy_ok(input_path, sample_rate):
            t0 = time.perf_counter()
            shutil.copy2(input_path, output_path)
            dur = time.perf_counter() - t0
            logger.info("Copied %s -> %s in %.2fs", input_path, output_path, dur)
            return output_path, True, None
    except Exception as e:
        # Fall through to re-encode on any probe/copy errors
        logger.debug("WAV copy check failed for %s: %s", input_path, e)

    # Build ffmpeg command optimized for speed
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-c:a",
        codec,
        "-threads",
        "0",
        str(output_path),
    ]

    try:
        t0 = time.perf_counter()
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        dur = time.perf_counter() - t0
        logger.info("Converted %s -> %s in %.2fs", input_path, output_path, dur)
        return output_path, True, None
    except subprocess.CalledProcessError as e:
        logger.error("ffmpeg failed for %s: %s", input_path, e)
        return output_path, False, str(e)
    except Exception as e:  # pragma: no cover - unexpected
        logger.error("Unexpected error converting %s: %s", input_path, e)
        return output_path, False, str(e)


def _gather_input_files(paths: Iterable[str], extensions: Optional[set] = None) -> List[Path]:
    extensions = {e.lower() for e in (extensions or DEFAULT_EXTENSIONS)}
    files: List[Path] = []
    for p in paths:
        pth = Path(p)
        if pth.is_dir():
            for f in pth.rglob("*"):
                if f.is_file() and f.suffix.lower().lstrip(".") in extensions:
                    files.append(f)
        elif pth.is_file():
            files.append(pth)
    return sorted(files)


def convert_many(
    inputs: Iterable[str],
    output_dir: Optional[str] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    workers: Optional[int] = None,
    overwrite: bool = False,
) -> List[Tuple[Path, bool, Optional[str]]]:
    """Convert many files in parallel.

    Returns list of (output_path, success, error)
    """
    files = _gather_input_files(inputs)
    if not files:
        logger.warning("No input files found for: %s", inputs)
        return []

    output_dir = Path(output_dir) if output_dir else None
    tasks: List[Tuple[Path, Path]] = []
    for f in files:
        if output_dir:
            out = output_dir / (f.stem + ".wav")
        else:
            # place next to input with .wav suffix
            out = f.with_suffix(".wav")
        tasks.append((f, out))

    max_workers = workers or min(32, (os.cpu_count() or 1) * 2)
    results: List[Tuple[Path, bool, Optional[str]]] = []

    iterator = tasks
    use_tqdm = tqdm is not None

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(convert_file, inp, out, sample_rate, "pcm_s16le", overwrite): (inp, out) for inp, out in iterator}
        if use_tqdm:
            pbar = tqdm(total=len(futures), desc="converting", unit="file")
        try:
            for fut in as_completed(futures):
                inp, out = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    logger.exception("Conversion raised for %s", inp)
                    result = (out, False, str(e))
                results.append(result)
                if use_tqdm:
                    pbar.update(1)
        finally:
            if use_tqdm:
                pbar.close()
    total = time.perf_counter() - start
    logger.info("Converted %d files in %.2fs (avg %.2fs/file)", len(results), total, (total / len(results) if results else 0))
    return results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast, parallel audio conversion to WAV (16kHz, mono)")
    p.add_argument("inputs", nargs="+", help="Input files and/or directories")
    p.add_argument("-o", "--output-dir", help="Directory to write outputs (keeps names)" )
    p.add_argument("-w", "--workers", type=int, help="Number of parallel workers (default: auto)")
    p.add_argument("-r", "--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Output sample rate (default: 16000)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--quiet", action="store_true", help="Quiet logging")
    return p.parse_args()


def main_cli() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    results = convert_many(args.inputs, output_dir=out_dir, sample_rate=args.sample_rate, workers=args.workers, overwrite=args.overwrite)

    succ = sum(1 for _, ok, _ in results if ok)
    fail = len(results) - succ

    logger.info("Done: %d succeeded, %d failed", succ, fail)
    if fail > 0:
        logger.info("Failed files:")
        for out, ok, err in results:
            if not ok:
                logger.info("  %s -> %s", out, err)
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main_cli())
