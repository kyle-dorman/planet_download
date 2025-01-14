import numpy as np
from numpy.ma.core import MaskedArray


def contrast_stretch(image: np.ndarray | MaskedArray, p_low: int = 2, p_high: int = 98) -> np.ndarray:
    """Perform contrast stretching using percentiles."""
    if isinstance(image, MaskedArray):
        v_min, v_max = np.percentile(image.compressed(), (p_low, p_high))
    else:
        v_min, v_max = np.percentile(image, (p_low, p_high))

    return np.clip((image - v_min) / (v_max - v_min), 0, 1)


def calculate_zoom_level(bounds: list[float], map_width: int = 800) -> int:
    """
    Calculate an approximate zoom level based on bounding box size.
    :param bounds: [minx, miny, maxx, maxy]
    :param map_width: Width of the map in pixels (default 800).
    :return: Integer zoom level.
    """
    minx, miny, maxx, maxy = bounds
    width_degrees = maxx - minx
    world_degrees = 360  # Total degrees longitude in the world

    # Calculate zoom based on ratio of bounding box width to world width
    zoom = int(round((map_width / 256) / (width_degrees / world_degrees)).bit_length() - 2)
    return max(1, min(zoom, 18))  # Ensure zoom is between 1 and 18
