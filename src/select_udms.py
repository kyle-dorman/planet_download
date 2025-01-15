import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf
from shapely import Polygon

from src.config import DownloadConfig
from src.grid import (
    create_polygon_aligned_profile_update,
    find_most_common_crs,
    open_and_convert_grid,
    reproject_and_crop_to_grid,
)
from src.util import setup_logger

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


# Finds the best set of UDMs to satisfy a target coverage value for a grid region.
# Returns the image ids.
def filter_image_set(
    results_grid_dir: Path,
    grid_path: Path,
    config: DownloadConfig,
) -> list[str]:
    udm_paths = [pth for pth in (results_grid_dir / "udm").iterdir() if pth.suffix == ".tif"]

    # Choose CRS to work in
    crs = find_most_common_crs(udm_paths)

    # Load Target Grid and convert grid to udm crs
    grid = open_and_convert_grid(grid_path, crs)

    # Create the new consistent grid which all UDMs wil be cropped to
    profile_update = create_polygon_aligned_profile_update(grid, crs, config.ground_sample_distance)

    # Save reprojected and cropped intermediates
    temp_udm_dir = results_grid_dir / "udm_temp"
    temp_udm_dir.mkdir(exist_ok=True)

    # Crop the UDMs
    logger.debug("Cropping UDMs & calculating coverage")
    cropped_images = []
    coverages = []
    for udm_path in udm_paths:
        temp_path = temp_udm_dir / udm_path.name

        # Get the UDM in the consistent grid. Do not retain the intermediates.
        clipped_image = reproject_and_crop_to_grid(
            tif_path=udm_path,
            grid_geom=grid,
            profile_update=profile_update,
            repro_path=temp_path,
            out_path=None,
            channels=1,
        )
        cropped_images.append(clipped_image)
        coverage = calculate_mask_coverage(
            clipped_image,
            grid,
            config.ground_sample_distance,
        )
        coverages.append(coverage)

    # Use udms in most to least coverage order
    coverage_order = np.argsort(coverages)[::-1]

    # Find UDMs which improve overall grid coverage and add to the list

    # A grid of coverage counters
    coverage_count = np.zeros((profile_update["height"], profile_update["width"]), dtype=np.int32)
    include_images = []
    # Area covered by the target grid
    grid_pixel_area = grid.area / config.ground_sample_distance**2
    for idx in coverage_order:
        # Find areas where we there are valid pixels
        image = cropped_images[idx]
        valid_pixels = image[0] == 1

        # Areas that still need pixels
        to_add = coverage_count < config.coverage_count

        # Find the area of the image that still needs updates and could be updated by this image
        should_update = np.logical_and(valid_pixels, to_add)

        # Determine how much of the image counts would be imporoved by this image
        pct_adding = should_update.sum() / grid_pixel_area

        # Must add a resonable % of pixels to be included.
        if pct_adding > config.percent_added:
            coverage_count += should_update
            include_images.append(udm_paths[idx].stem)

    return include_images


@click.command()
@click.option("-c", "--config-file", type=click.Path(exists=True), required=True)
@click.option("-m", "--month", type=int)
@click.option("-y", "--year", type=int)
def main(
    config_file: Path,
    month: int,
    year: int,
):
    config_file = Path(config_file)
    base_config = OmegaConf.structured(DownloadConfig)
    override_config = OmegaConf.load(config_file)
    config: DownloadConfig = OmegaConf.merge(base_config, override_config)  # type: ignore

    save_path = config.save_dir / str(year) / str(month).zfill(2)
    save_path.mkdir(exist_ok=True, parents=True)

    # Save the configuration to a YAML file
    OmegaConf.save(config, save_path / "config.yaml")

    setup_logger(logger, save_path, log_filename="select_udms.log")

    logger.info(f"Selecting best UDMs for year={year} month={month} grids={config.grid_dir} to={save_path}")

    for grid_path in config.grid_dir.iterdir():
        results_grid_dir = save_path / grid_path.stem
        grid_udm_dir = results_grid_dir / "udm"
        if not grid_udm_dir.exists():
            logger.warning(f"No udms for {grid_path.stem}")
            continue

        csv_path = results_grid_dir / "images_to_download.csv"
        if csv_path.exists():
            logger.info(f"Download list exists for {grid_path.stem}. Skipping...")
            continue

        image_ids = filter_image_set(results_grid_dir, grid_path, config)

        # strip the _3B_udm2 from the file name
        # e.g. 20230901_182511_53_2486_3B_udm2.tif
        image_ids = ["_".join(image_id.split("_")[:4]) for image_id in image_ids]
        df = pd.DataFrame(image_ids)
        df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
