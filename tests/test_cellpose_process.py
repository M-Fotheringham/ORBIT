import unittest
from unittest.mock import Mock, patch

from orbit.models import cellpose_process


class _RecordingConnection:
    def __init__(self):
        self.messages = []
        self.closed = False

    def send(self, message):
        self.messages.append(message)

    def close(self):
        self.closed = True


class CellposeChildProcessTests(unittest.TestCase):
    def test_child_forwards_progress_and_result(self):
        connection = _RecordingConnection()
        request = {
            "image_paths": ["image.tif"],
            "marker_names": ["CD8"],
            "pixel_size_um": 0.5,
            "nuclear_channel_name": "DAPI",
            "membrane_width_um": 1.0,
            "fov_size": 1024,
            "fov_overlap": 0.2,
            "dapi_positive_only": True,
            "minimum_dapi_fraction": 0.01,
        }

        def segment(*_args, progress_callback, **_kwargs):
            progress_callback({"phase": "loading_model"})
            return [{"cell_count": 3}]

        with patch(
            "orbit.models.cellpose_segmentation.segment_project_image_paths",
            side_effect=segment,
        ):
            cellpose_process._segmentation_process_main(connection, request)

        self.assertEqual(
            connection.messages,
            [
                ("progress", {"phase": "loading_model"}),
                ("result", [{"cell_count": 3}]),
            ],
        )
        self.assertTrue(connection.closed)

    def test_child_returns_traceback_on_python_exception(self):
        connection = _RecordingConnection()
        request = {
            "image_paths": [],
            "marker_names": [],
            "pixel_size_um": 0.5,
            "nuclear_channel_name": None,
            "membrane_width_um": 1.0,
            "fov_size": 1024,
            "fov_overlap": 0.2,
            "dapi_positive_only": True,
            "minimum_dapi_fraction": 0.01,
        }

        with patch(
            "orbit.models.cellpose_segmentation.segment_project_image_paths",
            side_effect=ValueError("bad input"),
        ):
            cellpose_process._segmentation_process_main(connection, request)

        self.assertEqual(connection.messages[0][0], "error")
        self.assertIn("ValueError: bad input", connection.messages[0][1])
        self.assertTrue(connection.closed)

    def test_cuda_detection_returns_result(self):
        connection = _RecordingConnection()
        with patch(
            "orbit.models.cellpose_segmentation.cuda_compatibility_details",
            Mock(return_value={"available": True, "device_name": "TITAN RTX"}),
        ):
            cellpose_process._cuda_detection_process_main(connection)

        self.assertEqual(
            connection.messages,
            [("result", {"available": True, "device_name": "TITAN RTX"})],
        )
        self.assertTrue(connection.closed)


class _ReceiveConnection:
    def __init__(self, messages):
        self.messages = list(messages)
        self.closed = False

    def poll(self, _timeout=None):
        return bool(self.messages)

    def recv(self):
        return self.messages.pop(0)

    def close(self):
        self.closed = True


class _SendConnection:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _Process:
    def __init__(self, exitcode=0):
        self.exitcode = exitcode
        self.started = False
        self.terminated = False

    def start(self):
        self.started = True

    def is_alive(self):
        return False

    def join(self, timeout=None):
        pass

    def terminate(self):
        self.terminated = True


class _Context:
    def __init__(self, messages, exitcode=0):
        self.receive = _ReceiveConnection(messages)
        self.send = _SendConnection()
        self.process = _Process(exitcode=exitcode)

    def Pipe(self, duplex=False):
        if duplex is not False:
            raise AssertionError("The worker pipe must be one-way.")
        return self.receive, self.send

    def Process(self, **_kwargs):
        return self.process


class CellposeProcessSupervisorTests(unittest.TestCase):
    def test_supervisor_forwards_progress_and_returns_result(self):
        context = _Context(
            [
                ("progress", {"phase": "loading_model"}),
                ("result", [{"cell_count": 4}]),
            ]
        )
        progress = []

        with patch.object(
            cellpose_process.multiprocessing,
            "get_context",
            return_value=context,
        ):
            result = cellpose_process._run_process(
                object(), progress_callback=progress.append
            )

        self.assertEqual(progress, [{"phase": "loading_model"}])
        self.assertEqual(result, [{"cell_count": 4}])
        self.assertTrue(context.process.started)
        self.assertTrue(context.receive.closed)
        self.assertTrue(context.send.closed)

    def test_supervisor_reports_native_process_exit(self):
        context = _Context([], exitcode=3221225477)

        with (
            patch.object(
                cellpose_process.multiprocessing,
                "get_context",
                return_value=context,
            ),
            self.assertRaisesRegex(
                cellpose_process.CellposeProcessError,
                "exit code 3221225477",
            ),
        ):
            cellpose_process._run_process(object())


if __name__ == "__main__":
    unittest.main()
