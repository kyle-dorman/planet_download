import json
import logging
import os
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.mask import mask as rmask
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_geom
from shapely.geometry import Polygon, shape

logger = logging.getLogger(__name__)


# Calculates the coverage of a valid mask (e.g., cloud cover) over a specified
# geometry. Returns the percentage of the area covered by the mask.
def calculate_mask_coverage(image: np.ndarray, grid_geom: Polygon, ground_sample_distance: float) -> float:
    ground_sample_area = ground_sample_distance**2
    grid_area = grid_geom.area

    # calculate the total number of valid pixels and total valid area
    total_1s = (image[0] == 1).sum()
    total_valid_area = total_1s * ground_sample_area

    # Calculate the valid area coverage percent
    pct_coverage = total_valid_area / max(1, grid_area)

    return pct_coverage


# Calculates the pct of the grid that is covered by an image asset.
def calculate_intersection_pct(grid_geom: Polygon, asset_geom: Polygon) -> float:
    # https://github.com/shapely/shapely/issues/1345
    with np.errstate(invalid="ignore"):
        if not grid_geom.intersects(asset_geom):
            return 0.0
        intersection = grid_geom.intersection(asset_geom)
        return intersection.area / grid_geom.area


# Load the geojson grid.
def load_grid(grid_path: Path) -> Polygon:
    # Load Target Grid
    with open(grid_path) as f:
        grid_geojson = json.load(f)
        geom = grid_geojson["features"][0]["geometry"]
        grid_geom: Polygon = shape(geom)  # type: ignore

    return grid_geom


# Convert a geojson poygon to a different crs.
def open_and_convert_grid(grid_path: Path, crs: CRS) -> Polygon:
    # Load Target Grid
    with open(grid_path) as f:
        grid_geojson = json.load(f)
        geom = grid_geojson["features"][0]["geometry"]
        grid_geom: Polygon = shape(geom)  # type: ignore
        grid_crs = CRS.from_string(grid_geojson["crs"]["properties"]["name"])

    # Convert grid bounds to base udm crs
    grid_transformed: Polygon = shape(transform_geom(grid_crs, crs, grid_geom))  # type: ignore

    return grid_transformed


# Find the most common CRS from a list of .tif files
def find_most_common_crs(tif_paths: list[Path]) -> CRS:
    counter = Counter()
    # Use most commont EPSG
    for udm_path in tif_paths:
        with rasterio.open(udm_path) as src:
            crs = src.crs
            counter.update([crs.to_string()])

    most_common_crs_str, count = counter.most_common(1)[0]
    most_common_crs = CRS.from_string(most_common_crs_str)
    logger.info(f"{count}/{len(tif_paths)} EPSGs are {most_common_crs_str}")

    return most_common_crs


# Create a profile update neecessary to reproject and crop a raster to a polygon.
# Updates the CRS, transform and width/height.
def create_polygon_aligned_profile_update(grid: Polygon, crs: CRS, res: float) -> dict:
    # Get the bounding box of the polygon
    minx, miny, maxx, maxy = grid.bounds

    # Calculate the aligned extent (snap to the grid)
    aligned_minx = np.floor(minx / res) * res
    aligned_miny = np.floor(miny / res) * res
    aligned_maxx = np.ceil(maxx / res) * res
    aligned_maxy = np.ceil(maxy / res) * res

    # Define output width and height
    width = int((aligned_maxx - aligned_minx) / res)
    height = int((aligned_maxy - aligned_miny) / res)

    dst_transform = from_origin(aligned_minx, aligned_maxy, res, res)
    profile_update = {
        "crs": crs,
        "transform": dst_transform,
        "width": width,
        "height": height,
    }

    return profile_update


# Crop and reproject a raster to a specific grid polygon and return the result.
# Optionally, save results to a file.
def reproject_and_crop_to_grid(
    tif_path: Path,
    grid_geom: Polygon,
    profile_update: dict,
    repro_path: Path,
    out_path: Path | None,
    channels: int | None,
) -> np.ndarray:
    # open the geotiff file
    with rasterio.open(tif_path) as src:
        # update the raster profile
        dst_profile = src.profile.copy()
        dst_profile.update(profile_update)

        if channels is None:
            channels = src.count
        assert channels is not None

        # Only reproject certain bands
        if channels != src.count:
            dst_profile.update(
                {
                    "count": channels,  # First band only,
                }
            )

        # Create empty array for the output
        reprojected_array = np.empty((channels, profile_update["height"], profile_update["width"]), dtype=src.dtypes[0])

        # Reproject bands
        for band in range(1, channels + 1):
            reproject(
                source=rasterio.band(src, band),
                destination=reprojected_array[band - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=profile_update["transform"],
                dst_crs=profile_update["crs"],
                resampling=Resampling.bilinear,
            )

        # Write the reprojected raster
        with rasterio.open(repro_path, "w", **dst_profile) as repro_dst:
            repro_dst.write(reprojected_array)

    # Mask the raster using the polygon.
    # Since it is already reprojected, we just have to mask out pixels.
    with rasterio.open(repro_path) as src:
        clipped_image, _ = rmask(src, [grid_geom], crop=False)

    # Write the final clipped raster if saving outputs, otherwise delete the reprojected intermediate.
    if out_path is not None:
        with rasterio.open(out_path, "w", **dst_profile) as clipped_dst:
            clipped_dst.write(clipped_image)
    else:
        os.remove(repro_path)

    return clipped_image
