"""Run CUDA-backed Cellpose work outside the Napari/VisPy process."""

from __future__ import annotations

import multiprocessing
import traceback
from collections.abc import Callable, Iterable
from pathlib import Path


_POLL_INTERVAL_SECONDS = 0.1
_SHUTDOWN_TIMEOUT_SECONDS = 5.0


class CellposeProcessError(RuntimeError):
    """Raised when the isolated Cellpose process fails or exits unexpectedly."""


def _send_message(connection, kind: str, payload) -> None:
    connection.send((kind, payload))


def _segmentation_process_main(connection, request: dict) -> None:
    """Child-process entry point; deliberately imports no Qt or Napari modules."""
    try:
        from orbit.models.cellpose_segmentation import segment_project_image_paths

        def report_progress(update):
            _send_message(connection, "progress", update)

        results = segment_project_image_paths(
            request["image_paths"],
            request["marker_names"],
            pixel_size_um=request["pixel_size_um"],
            progress_callback=report_progress,
            nuclear_channel_name=request["nuclear_channel_name"],
            membrane_width_um=request["membrane_width_um"],
            fov_size=request["fov_size"],
            fov_overlap=request["fov_overlap"],
            dapi_positive_only=request["dapi_positive_only"],
            minimum_dapi_fraction=request["minimum_dapi_fraction"],
        )
        _send_message(connection, "result", results)
    except BaseException:
        _send_message(connection, "error", traceback.format_exc())
    finally:
        connection.close()


def _cuda_detection_process_main(connection) -> None:
    """Detect CUDA in a child so the GUI never initializes the CUDA runtime."""
    try:
        from orbit.models.cellpose_segmentation import cuda_compatibility_details

        _send_message(connection, "result", cuda_compatibility_details())
    except BaseException:
        _send_message(connection, "error", traceback.format_exc())
    finally:
        connection.close()


def _run_process(target, args=(), progress_callback=None):
    context = multiprocessing.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=target,
        args=(send_connection, *args),
        name="ORBIT Cellpose worker",
    )
    try:
        process.start()
    except Exception:
        receive_connection.close()
        send_connection.close()
        raise
    send_connection.close()

    try:
        while True:
            if receive_connection.poll(_POLL_INTERVAL_SECONDS):
                try:
                    kind, payload = receive_connection.recv()
                except EOFError:
                    process.join()
                    raise CellposeProcessError(
                        "The isolated Cellpose process closed without returning "
                        f"a result (exit code {process.exitcode})."
                    )
                if kind == "progress":
                    if progress_callback is not None:
                        progress_callback(payload)
                    continue
                if kind == "result":
                    return payload
                if kind == "error":
                    raise CellposeProcessError(payload)
                raise CellposeProcessError(
                    f"Cellpose worker returned an unknown message: {kind!r}"
                )

            if not process.is_alive():
                process.join()
                raise CellposeProcessError(
                    "The isolated Cellpose process stopped unexpectedly "
                    f"(exit code {process.exitcode}). The ORBIT window remains "
                    "open; retry after checking the NVIDIA driver and input image."
                )
    finally:
        receive_connection.close()
        process.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
        if process.is_alive():
            process.terminate()
            process.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)


def cuda_compatibility_details_isolated() -> dict:
    """Return CUDA diagnostics without loading CUDA libraries in the GUI."""
    return dict(_run_process(_cuda_detection_process_main))


def segment_project_image_paths_isolated(
    image_paths: Iterable[str | Path],
    selected_marker_names: Iterable[str],
    pixel_size_um: float,
    progress_callback: Callable[[dict], None] | None = None,
    *,
    nuclear_channel_name: str | None,
    membrane_width_um: float,
    fov_size: int,
    fov_overlap: float,
    dapi_positive_only: bool,
    minimum_dapi_fraction: float,
) -> list[dict]:
    """Segment images in a spawned process isolated from Qt and OpenGL."""
    request = {
        "image_paths": [str(path) for path in image_paths],
        "marker_names": [str(name) for name in selected_marker_names],
        "pixel_size_um": float(pixel_size_um),
        "nuclear_channel_name": (
            None if nuclear_channel_name is None else str(nuclear_channel_name)
        ),
        "membrane_width_um": float(membrane_width_um),
        "fov_size": int(fov_size),
        "fov_overlap": float(fov_overlap),
        "dapi_positive_only": bool(dapi_positive_only),
        "minimum_dapi_fraction": float(minimum_dapi_fraction),
    }
    return _run_process(
        _segmentation_process_main,
        (request,),
        progress_callback=progress_callback,
    )


__all__ = [
    "CellposeProcessError",
    "cuda_compatibility_details_isolated",
    "segment_project_image_paths_isolated",
]
