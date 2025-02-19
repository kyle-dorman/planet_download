import logging
import tempfile
from datetime import datetime
from pathlib import Path

import click
import geopandas as gpd
import numpy as np
import pandas as pd

from src.config import DownloadConfig
from src.grid import (
    calculate_intersection_pct,
    calculate_mask_coverage,
    create_polygon_aligned_profile_update,
    find_most_common_crs,
    open_and_convert_grid,
    reproject_and_crop_to_grid,
)
from src.util import (
    cleaned_asset_id,
    create_config,
    geojson_paths,
    get_tqdm,
    is_notebook,
    parse_tif_path,
    setup_logger,
    tif_paths,
)

logger = logging.getLogger(__name__)


def update_coverage(
    coverage_order: list[int],
    coverages: list[tuple[np.ndarray, float, float, datetime]],
    udm_paths: list[Path],
    coverage_count: np.ndarray,
    grid_pixel_area: float,
    skip_same_day: bool,
    config: DownloadConfig,
) -> list[dict]:
    item_coverage = []
    dates_added = set()

    for idx in coverage_order:
        # Find areas where we there are valid pixels
        clipped_image, clear_coverage, intersection_pct, tif_datetime = coverages[idx]
        valid_pixels = clipped_image[0] == 1

        # Areas that still need pixels
        to_add = coverage_count < config.coverage_count

        # Find the area of the image that still needs updates and could be updated by this image
        should_update = np.logical_and(valid_pixels, to_add)

        # Determine how much of the image counts would be imporoved by this image
        pct_adding = should_update.sum() / grid_pixel_area
        skip_for_date = skip_same_day and tif_datetime.date() in dates_added
        include_image = pct_adding > config.percent_added and not skip_for_date

        if include_image:
            dates_added.add(tif_datetime.date())

        # Save stats for all UDMs
        if not skip_for_date:
            item_coverage.append(
                {
                    "ordered_idx": idx,
                    "asset_id": cleaned_asset_id(udm_paths[int(idx)]),
                    "clear_coverge_pct": clear_coverage,
                    "intersection_pct": intersection_pct,
                    "pct_adding": pct_adding,
                    "capture_datetime": tif_datetime,
                    "include_image": include_image,
                }
            )

        # Update counts for valid image pixels if the image will be inlcuded
        if include_image:
            coverage_count[to_add] += 1

    return item_coverage


# Finds the best set of UDMs to satisfy a target coverage value for a grid region.
# Returns a DataFrame of information about UDM coverage.
def calculate_udm_coverages(
    results_grid_dir: Path,
    grid_path: Path,
    config: DownloadConfig,
) -> pd.DataFrame:
    udm_paths = tif_paths(results_grid_dir / "udm")
    geojson_file = results_grid_dir / "search_geometries.geojson"
    gdf = gpd.read_file(geojson_file)

    # Choose CRS to work in
    crs = find_most_common_crs(udm_paths)

    # Load Target Grid and convert grid to udm crs
    grid = open_and_convert_grid(grid_path, crs)

    # Create the new consistent grid which all UDMs wil be cropped to
    profile_update = create_polygon_aligned_profile_update(grid, crs, config.ground_sample_distance)

    # Crop the UDMs
    logger.debug("Cropping UDMs & calculating coverage")
    coverages = []
    # Remove reprojected and cropped intermediates at the end
    with tempfile.TemporaryDirectory() as tempdir:
        for udm_path in udm_paths:
            temp_path = Path(tempdir) / udm_path.name
            tif_datetime = parse_tif_path(udm_path)

            # Get the UDM in the consistent grid. Do not retain the intermediates.
            clipped_image = reproject_and_crop_to_grid(
                tif_path=udm_path,
                grid_geom=grid,
                profile_update=profile_update,
                repro_path=temp_path,
                out_path=None,
                channels=1,
            )
            clear_coverage = calculate_mask_coverage(
                clipped_image,
                grid,
                config.ground_sample_distance,
            )
            item_geom = gdf[gdf.id == cleaned_asset_id(udm_path)].geometry.iloc[0]
            intersection_pct = calculate_intersection_pct(grid, item_geom)
            coverages.append((clipped_image, clear_coverage, intersection_pct, tif_datetime))

    # Use udms in most to least coverage order
    coverage_order = np.argsort([coverage for _, coverage, _, _ in coverages])[::-1].tolist()

    # Find UDMs which improve overall grid coverage

    # A grid of coverage counters
    coverage_count = np.zeros((profile_update["height"], profile_update["width"]), dtype=np.int32)
    # Area covered by the target grid
    grid_pixel_area = grid.area / config.ground_sample_distance**2

    item_coverage = update_coverage(
        coverage_order,
        coverages,
        udm_paths,
        coverage_count,
        grid_pixel_area,
        skip_same_day=config.skip_same_day,
        config=config,
    )
    included_item_idxes = {d["ordered_idx"] for d in item_coverage}
    skipped_coverage_order = [i for i in coverage_order if i not in included_item_idxes]
    skipped_item_coverage = update_coverage(
        skipped_coverage_order,
        coverages,
        udm_paths,
        coverage_count,
        grid_pixel_area,
        skip_same_day=False,
        config=config,
    )

    return pd.DataFrame(item_coverage + skipped_item_coverage)


def select_udms(
    config_file: Path,
    year: int,
    month: int,
) -> None:
    config, save_path = create_config(config_file, year=year, month=month)

    setup_logger(save_path, log_filename="select_udms.log")

    logger.info(f"Selecting best UDMs for year={year} month={month} grids={config.grid_dir} to={save_path}")

    in_notebook = is_notebook()

    tqdm = get_tqdm(use_async=False, in_notebook=in_notebook)

    for grid_path in tqdm(geojson_paths(config.grid_dir)):
        grid_id = grid_path.stem
        logger.debug(f"Selecting best UDMs for {grid_id}")

        results_grid_dir = save_path / grid_id
        grid_udm_dir = results_grid_dir / "udm"
        if not grid_udm_dir.exists():
            logger.warning(f"No udms for {grid_id}")
            continue

        csv_path = results_grid_dir / "images_to_download.csv"
        if csv_path.exists():
            logger.debug(f"Download list exists for {grid_id}. Skipping...")
            continue

        coverage_df = calculate_udm_coverages(results_grid_dir, grid_path, config)
        coverage_df.to_csv(csv_path, index=False)


@click.command()
@click.option("-c", "--config-file", type=click.Path(exists=True), required=True)
@click.option("-y", "--year", type=click.IntRange(min=1990, max=2050))
@click.option("-m", "--month", type=click.IntRange(min=1, max=12))
def main(
    config_file: Path,
    year: int,
    month: int,
):
    config_file = Path(config_file)

    select_udms(config_file=config_file, month=month, year=year)

    logger.info("Done!")


if __name__ == "__main__":
    main()
