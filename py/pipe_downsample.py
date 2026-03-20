# pipe_downsample.py
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import librosa
import soundfile as sf
import numpy as np

# --- robust local imports: loger.py + config_pipe.py next to this script ---
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from loger import setup_logging  # uses file+console handlers :contentReference[oaicite:2]{index=2}

try:
    import config_pipe  # provides config_pipe.env defaults
except Exception:
    config_pipe = None


def _pick_log_path() -> Path:
    """
    Priority:
      1) env LOG_FILE (so notebook controls it)
      2) config_pipe.env["LOG_FILE"] if available
      3) cwd/audio_processing.log
    """
    lf = os.environ.get("LOG_FILE", "").strip()
    if lf:
        return Path(lf).expanduser().resolve()

    if config_pipe is not None and hasattr(config_pipe, "env"):
        lf2 = str(config_pipe.env.get("LOG_FILE", "")).strip()
        if lf2:
            return Path(lf2).expanduser().resolve()

    return (Path.cwd() / "audio_processing.log").resolve()


def _configure_logger(quiet: bool) -> logging.Logger:
    """
    Always log INFO to file. Console is INFO unless --quiet, then WARNING+.
    """
    log_path = _pick_log_path()
    logger = setup_logging(log_file=str(log_path), level=logging.INFO)

    if quiet:
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                h.setLevel(logging.WARNING)

    logger.info(f"Log file: {log_path}")
    return logger


def build_output_path(out_dir: Path, in_file: Path, target_sr: int) -> Path:
    # clip1.mp3 -> clip1_16khz.flac
    return out_dir / f"{in_file.stem}_{int(target_sr/1000)}khz.flac"


def find_inputs(audio_dir: Path, pattern: str, recursive: bool = False) -> List[Path]:
    if recursive:
        return sorted(audio_dir.rglob(pattern))
    return sorted(audio_dir.glob(pattern))


def _resample_if_needed(y: np.ndarray, sr: int, target_sr: int, logger: logging.Logger) -> Tuple[np.ndarray, int]:
    """
    y shape is (channels, samples) or (samples,)
    returns (y_out, sr_out)
    """
    if y.ndim == 1:
        y = y[None, :]

    if sr > target_sr:
        logger.info(f"Resampling {sr} Hz -> {target_sr} Hz")
        y_out = np.stack(
            [librosa.resample(y[ch], orig_sr=sr, target_sr=target_sr) for ch in range(y.shape[0])],
            axis=0,
        )
        return y_out, target_sr

    if sr < target_sr:
        logger.warning(f"Input SR {sr} Hz < target {target_sr} Hz. No upsampling performed.")
    else:
        logger.info(f"Already at target SR {target_sr} Hz. No resampling needed.")

    return y, sr


def downsample_audio(
    input_path: Path,
    output_path: Path,
    target_sr: int,
    logger: logging.Logger,
    force: bool = False,
    skip_existing: bool = True,
) -> str:
    """
    Returns status: "ok" | "skipped_exists" | "skipped_duplicate" | "fail"
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and output_path.exists() and output_path.stat().st_size > 0 and not force:
        logger.info(f"[SKIP exists] {output_path}")
        return "skipped_exists"

    logger.info(f"Loading audio: {input_path}")
    y, sr = librosa.load(str(input_path), sr=None, mono=False)

    y_out, sr_out = _resample_if_needed(y, sr, target_sr, logger)

    # soundfile expects (samples, channels) for multi-channel
    if y_out.shape[0] > 1:
        sf.write(str(output_path), y_out.T, sr_out, format="FLAC")
    else:
        sf.write(str(output_path), y_out.reshape(-1), sr_out, format="FLAC")

    logger.info(f"[OK] Wrote: {output_path}")
    return "ok"


def run_batch(
    audio_dir: Path,
    out_dir: Path,
    pattern: str = "clip*.mp3",
    target_sr: int = 16000,
    quiet: bool = False,
    recursive: bool = False,
    skip_existing: bool = True,
    force: bool = False,
    log_file: Optional[str] = None,
) -> Dict[str, int]:
    """
    Jupyter-friendly entrypoint (no argparse, no env required).
    """
    if log_file is not None:
        os.environ["LOG_FILE"] = str(Path(log_file).expanduser().resolve())

    logger = _configure_logger(quiet)

    audio_dir = Path(audio_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Downsample start (run_batch) ===")
    logger.info(f"cwd={Path.cwd()}")
    logger.info(f"AUDIOFOLDER={audio_dir}")
    logger.info(f"OUTFOLDER={out_dir}")
    logger.info(f"pattern={pattern}")
    logger.info(f"target_sr={target_sr}")
    logger.info(f"skip_existing={skip_existing} force={force} recursive={recursive}")

    inputs = find_inputs(audio_dir, pattern, recursive=recursive)
    logger.info(f"Total matched files: {len(inputs)}")

    # Prevent duplicates caused by collisions to the same output name
    seen_outputs = set()

    counts = {"ok": 0, "skipped_exists": 0, "skipped_duplicate": 0, "fail": 0}

    for f in inputs:
        try:
            out_path = build_output_path(out_dir, f, target_sr)

            if out_path in seen_outputs:
                logger.warning(f"[SKIP duplicate->same output] in={f} out={out_path}")
                counts["skipped_duplicate"] += 1
                continue
            seen_outputs.add(out_path)

            status = downsample_audio(
                input_path=f,
                output_path=out_path,
                target_sr=target_sr,
                logger=logger,
                force=force,
                skip_existing=skip_existing,
            )
            counts[status] += 1
        except Exception as e:
            counts["fail"] += 1
            logger.exception(f"[FAIL] {f}: {e}")

    logger.info("=== Downsample summary ===")
    for k, v in counts.items():
        logger.info(f"{k}: {v}")
    logger.info("=== Downsample finished ===")

    return counts


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Downsample audio files to a target sample rate.")
    ap.add_argument("--pattern", type=str, default="*.wav", help="Glob pattern within AUDIOFOLDER.")
    ap.add_argument("--target_sr", type=int, default=16000, help="Target sample rate (Hz).")
    ap.add_argument("--quiet", action="store_true", help="Reduce console output; keep file logs.")
    ap.add_argument("--recursive", action="store_true", help="Use recursive glob (rglob) instead of glob.")
    ap.add_argument("--skip_existing", action="store_true", default=True, help="Skip if output exists (default: on).")
    ap.add_argument("--force", action="store_true", help="Overwrite outputs (disables skip_existing).")

    args = ap.parse_args(argv)

    # In CLI: if --force, treat skip_existing as False
    if args.force:
        args.skip_existing = False

    logger = _configure_logger(args.quiet)

    audio_dir = Path(os.environ.get("AUDIOFOLDER", "")).expanduser()
    out_dir = Path(os.environ.get("OUTFOLDER", "")).expanduser()

    if not str(audio_dir):
        logger.error("Missing env var AUDIOFOLDER.")
        return 2
    if not str(out_dir):
        logger.error("Missing env var OUTFOLDER.")
        return 2
    if not audio_dir.exists():
        logger.error(f"AUDIOFOLDER does not exist: {audio_dir}")
        return 2

    counts = run_batch(
        audio_dir=audio_dir,
        out_dir=out_dir,
        pattern=args.pattern,
        target_sr=args.target_sr,
        quiet=args.quiet,
        recursive=args.recursive,
        skip_existing=args.skip_existing,
        force=args.force,
        log_file=None,  # uses LOG_FILE env/config/cwd
    )

    return 0 if counts["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
#     raise SystemExit(main())
