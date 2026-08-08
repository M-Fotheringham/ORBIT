import numpy as np


DEFAULT_MINIMUM_DAPI_FRACTION = 0.01
DEFAULT_MAXIMUM_RANDOM_ATTEMPTS = 200
OVERVIEW_THRESHOLD_SIZE = 768


class RandomFOVGenerator:
    def __init__(self, qptiff_image):
        self.qptiff_image = qptiff_image
        self._dapi_threshold_cache = {}

    def random_position(
        self,
        size: int = 512,
        seed: int | None = None,
        dapi_channel: int | None = None,
        minimum_dapi_fraction: float = DEFAULT_MINIMUM_DAPI_FRACTION,
        maximum_attempts: int = DEFAULT_MAXIMUM_RANDOM_ATTEMPTS,
    ):
        rng = np.random.default_rng(seed)

        _n_channels, height, width = self.qptiff_image.get_shape()

        if size > height or size > width:
            raise ValueError(
                f"FOV size {size} is larger than image dimensions {(height, width)}"
            )
        if not 0.0 <= float(minimum_dapi_fraction) <= 1.0:
            raise ValueError("minimum_dapi_fraction must be between 0 and 1.")
        if maximum_attempts <= 0:
            raise ValueError("maximum_attempts must be positive.")

        maximum_y0 = height - size
        maximum_x0 = width - size
        if dapi_channel is None:
            y0 = int(rng.integers(0, maximum_y0 + 1))
            x0 = int(rng.integers(0, maximum_x0 + 1))
            return y0, x0

        threshold = self._dapi_threshold(dapi_channel)
        best_fraction = 0.0
        for _attempt in range(int(maximum_attempts)):
            y0 = int(rng.integers(0, maximum_y0 + 1))
            x0 = int(rng.integers(0, maximum_x0 + 1))
            dapi_fov = self.get_fov(
                y0=y0,
                x0=x0,
                size=size,
                channel=dapi_channel,
            )
            dapi_fraction = self.dapi_positive_fraction(
                dapi_fov,
                threshold,
            )
            best_fraction = max(best_fraction, dapi_fraction)
            if dapi_fraction >= float(minimum_dapi_fraction):
                return y0, x0

        raise ValueError(
            "Could not find a DAPI-positive field after "
            f"{maximum_attempts} attempts. The best candidate contained "
            f"{best_fraction:.2%} DAPI-positive pixels; "
            f"{minimum_dapi_fraction:.2%} is required."
        )

    def _dapi_threshold(self, dapi_channel: int) -> float:
        dapi_channel = int(dapi_channel)
        if dapi_channel in self._dapi_threshold_cache:
            return self._dapi_threshold_cache[dapi_channel]

        overview = np.asarray(
            self.qptiff_image.get_overview(
                channel=dapi_channel,
                max_size=OVERVIEW_THRESHOLD_SIZE,
            )
        )
        finite = overview[np.isfinite(overview)]
        if finite.size == 0:
            raise ValueError("The DAPI channel contains no finite pixels.")
        background = float(np.percentile(finite, 20.0))
        bright = float(np.percentile(finite, 99.5))
        if bright <= background:
            # Very sparse nuclear signal can occupy less than the upper 0.5%
            # of a whole-slide overview. Preserve that valid case by falling
            # back to the brightest finite pixel.
            bright = float(np.max(finite))
        if (
            not np.isfinite(background)
            or not np.isfinite(bright)
            or bright <= background
        ):
            raise ValueError(
                "The DAPI channel has no measurable intensity range, so a "
                "DAPI-positive field cannot be selected."
            )
        threshold = background + 0.20 * (bright - background)
        self._dapi_threshold_cache[dapi_channel] = threshold
        return threshold

    @staticmethod
    def dapi_positive_fraction(dapi_fov: np.ndarray, threshold: float) -> float:
        values = np.asarray(dapi_fov)
        finite = np.isfinite(values)
        if not np.any(finite):
            return 0.0
        return float(np.count_nonzero(values[finite] > threshold) / finite.sum())

    def get_fov(
        self,
        y0: int,
        x0: int,
        size: int = 512,
        channel: int = 0,
    ):
        return self.qptiff_image.get_region(
            channel=channel,
            y0=y0,
            x0=x0,
            height=size,
            width=size,
        )


__all__ = [
    "DEFAULT_MAXIMUM_RANDOM_ATTEMPTS",
    "DEFAULT_MINIMUM_DAPI_FRACTION",
    "RandomFOVGenerator",
]
