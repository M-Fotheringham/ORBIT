from pathlib import Path

import tifffile as tiff
from qptifffile import QPTiffFile


class QPTiffImage:
    def __init__(self, path: str | Path):
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(f"Could not find file: {self.path}")

        self.qptiff = QPTiffFile(self.path)
        self.channel_names = list(self.qptiff.get_biomarkers())

        self.tif = tiff.TiffFile(self.path)
        self.series = self.tif.series[0]

        self.shape = self.series.shape
        self.dtype = self.series.dtype

        n_channels = self.shape[0]

        self.channel_names = self.channel_names[:n_channels]

        if len(self.channel_names) < n_channels:
            self.channel_names += [
                f"Channel {i}" for i in range(len(self.channel_names), n_channels)
            ]

    def get_shape(self):
        return self.shape

    def get_channel_names(self):
        return self.channel_names

    def get_channel(self, channel: int = 0):
        return self.series.asarray(key=channel)
