"""Download and verify the Cellpose-SAM model used by ORBIT builds.

This script deliberately does not import Cellpose or PyTorch. It stages the
large model as a regular build input so PyInstaller can copy it into ORBIT's
standalone distribution and the installer can install it with the application.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path


MODEL_NAME = "cpsam_v2"
MODEL_REPOSITORY_REVISION = "7c61431b5fbb078f3296754bd15d9f51b320f837"
MODEL_URL = (
    "https://huggingface.co/mouseland/cellpose-sam/resolve/"
    f"{MODEL_REPOSITORY_REVISION}/{MODEL_NAME}"
)
MODEL_SIZE_BYTES = 1_233_586_851
MODEL_SHA256 = "0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667"
CHUNK_SIZE = 8 * 1024 * 1024
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = REPOSITORY_ROOT / "build" / "cellpose_models" / MODEL_NAME
DEFAULT_CELLPOSE_CACHE_MODEL = Path.home() / ".cellpose" / "models" / MODEL_NAME


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model(path: Path) -> None:
    """Raise when a staged model is incomplete or is not the pinned model."""
    if not path.is_file():
        raise FileNotFoundError(f"Cellpose model not found: {path}")
    actual_size = path.stat().st_size
    if actual_size != MODEL_SIZE_BYTES:
        raise ValueError(
            f"Invalid {MODEL_NAME} size: expected {MODEL_SIZE_BYTES:,} bytes, "
            f"found {actual_size:,} bytes."
        )
    actual_sha256 = file_sha256(path)
    if actual_sha256.lower() != MODEL_SHA256:
        raise ValueError(
            f"Invalid {MODEL_NAME} SHA-256: expected {MODEL_SHA256}, "
            f"found {actual_sha256}."
        )


def _print_progress(downloaded: int, total: int) -> None:
    percent = min(downloaded / total * 100.0, 100.0) if total else 0.0
    print(
        f"\rStaging {MODEL_NAME}: {downloaded / 1024**2:,.1f} / "
        f"{total / 1024**2:,.1f} MiB ({percent:5.1f}%)",
        end="",
        flush=True,
    )


def download_model(destination: Path, url: str = MODEL_URL) -> None:
    """Atomically download the pinned model and validate its exact bytes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ORBIT model staging"},
    )
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as out:
            expected = int(response.headers.get("Content-Length") or MODEL_SIZE_BYTES)
            downloaded = 0
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                _print_progress(downloaded, expected)
        print()
        validate_model(temporary)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def stage_model(destination: Path, source: Path | None = None) -> Path:
    """Reuse, copy, or download the verified model into the build directory."""
    destination = destination.expanduser().resolve()
    if destination.is_file():
        try:
            validate_model(destination)
        except ValueError:
            destination.unlink()
        else:
            print(f"Using verified staged model: {destination}")
            return destination

    if source is None and DEFAULT_CELLPOSE_CACHE_MODEL.is_file():
        try:
            validate_model(DEFAULT_CELLPOSE_CACHE_MODEL)
        except ValueError:
            print(
                "Ignoring an incompatible model in Cellpose's user cache: "
                f"{DEFAULT_CELLPOSE_CACHE_MODEL}"
            )
        else:
            source = DEFAULT_CELLPOSE_CACHE_MODEL
            print(f"Reusing Cellpose's verified cached model: {source}")

    if source is not None:
        source = source.expanduser().resolve()
        validate_model(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.part")
        temporary.unlink(missing_ok=True)
        try:
            shutil.copyfile(source, temporary)
            validate_model(temporary)
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    else:
        print(f"Downloading the pinned {MODEL_NAME} model once for this build...")
        download_model(destination)

    print(f"Staged verified model: {destination}")
    return destination


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage and verify cpsam_v2 for the ORBIT application build."
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=f"Build destination (default: {DEFAULT_DESTINATION})",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Use an existing cpsam_v2 file instead of downloading it.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the destination without downloading or copying.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    arguments = parse_arguments(argv)
    try:
        if arguments.check:
            validate_model(arguments.destination.expanduser().resolve())
            print(f"Verified staged model: {arguments.destination.resolve()}")
        else:
            stage_model(arguments.destination, source=arguments.source)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
