from pathlib import Path
import xml.etree.ElementTree as ET

import tifffile as tiff


class QPTiffImage:
    def __init__(self, path: str | Path):

        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(f"Could not find file: {self.path}")

        self.extension = self.path.suffix.lower()

        self.tif = tiff.TiffFile(self.path)
        self.series = self.tif.series[0]

        self.shape = self.series.shape
        self.dtype = self.series.dtype

        self.channel_names = self._get_channel_names()

    def get_shape(self):
        return self.shape

    def get_channel_names(self):
        return self.channel_names

    def get_channel(self, channel: int = 0):
        return self.series.asarray(key=channel)

    def _get_channel_names(self):

        n_channels = self.shape[0]

        # ============================================================
        # QPTIFF
        # ============================================================

        if self.extension == ".qptiff":

            try:
                from qptifffile import QPTiffFile

                qptiff = QPTiffFile(self.path)

                names = list(qptiff.get_biomarkers())

                if len(names) >= n_channels:
                    return names[:n_channels]

            except Exception:
                pass

        # ============================================================
        # OME-TIFF
        # ============================================================

        try:

            ome_xml = self.tif.ome_metadata

            if ome_xml is not None:

                root = ET.fromstring(ome_xml)

                namespaces = {
                    "ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"
                }

                channels = root.findall(".//ome:Channel", namespaces)

                names = [
                    ch.attrib.get("Name", f"Channel {i}")
                    for i, ch in enumerate(channels)
                ]

                if len(names) > 0:
                    return names[:n_channels]

        except Exception:
            pass

        # ============================================================
        # IMAGEJ TIFF
        # ============================================================

        try:

            metadata = self.tif.imagej_metadata

            if metadata is not None:

                labels = metadata.get("Labels")

                if labels is not None:

                    labels = list(labels)

                    if len(labels) > 0:
                        return labels[:n_channels]

        except Exception:
            pass

        # ============================================================
        # TIFF PAGE DESCRIPTIONS
        # ============================================================

        try:

            names = []

            for i, page in enumerate(self.tif.pages[:n_channels]):

                desc = str(page.description)

                if "Name=" in desc:

                    name = desc.split("Name=")[1].split("\n")[0].strip()

                elif "ChannelName=" in desc:

                    name = (
                        desc
                        .split("ChannelName=")[1]
                        .split("\n")[0]
                        .strip()
                    )

                else:
                    name = f"Channel {i}"

                names.append(name)

            if len(names) > 0:
                return names

        except Exception:
            pass

        # ============================================================
        # FALLBACK
        # ============================================================

        return [f"Channel {i}" for i in range(n_channels)]