import numpy as np


class RandomFOVGenerator:
    def __init__(self, qptiff_image):
        self.qptiff_image = qptiff_image

    def random_position(
        self,
        size: int = 512,
        seed: int | None = None,
    ):
        rng = np.random.default_rng(seed)

        n_channels, height, width = self.qptiff_image.get_shape()

        if size > height or size > width:
            raise ValueError(
                f"FOV size {size} is larger than image dimensions {(height, width)}"
            )

        y0 = rng.integers(0, height - size)
        x0 = rng.integers(0, width - size)

        return y0, x0

    def get_fov(
        self,
        y0: int,
        x0: int,
        size: int = 512,
        channel: int = 0,
    ):
        channel_img = self.qptiff_image.get_channel(channel)

        return channel_img[y0 : y0 + size, x0 : x0 + size]
