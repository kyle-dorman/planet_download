import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import click
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.errors import WarpOperationError
from shapely import Polygon

from src.config import CLOUD_BAND, CONFIDENCE_BAND, HEAVY_HAZE_BAND, LIGHT_HAZE_BAND, SHADOW_BAND, DownloadConfig
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
    is_within_n_hours,
    log_structured_failure,
    parse_acquisition_datetime,
    setup_logger,
    tif_paths,
)

import warnings  # isort: skip

warnings.filterwarnings(
    "ignore",
    message=".*errors='ignore'.*is deprecated.*",
    category=FutureWarning,
)

logger = logging.getLogger(__name__)
CATEGORY = Path(__file__).stem


def update_coverage(
    coverage_order: list[int],
    coverages: list[tuple[np.ndarray, float, float, datetime]],
    udm_paths: list[Path],
    coverage_count: np.ndarray,
    grid_pixel_area: float,
    skip_same_range_days: float,
    config: DownloadConfig,
) -> list[dict]:
    item_coverage = []
    dates_added = set()

    for idx in coverage_order:
        # Find areas where we there are valid pixels
        clipped_image, clear_coverage, intersection_pct, tif_datetime = coverages[idx]
        valid_pixels = clipped_image == 1

        # Areas that still need pixels
        to_add = coverage_count < config.coverage_count

        # Find the area of the image that still needs updates and could be updated by this image
        should_update = np.logical_and(valid_pixels, to_add)

        # Determine how much of the image counts would be imporoved by this image
        pct_adding = 100 * should_update.sum() / grid_pixel_area

        skip_for_date = skip_same_range_days > 0 and is_within_n_hours(
            tif_datetime, dates_added, n_hours=int(skip_same_range_days * 24)
        )
        include_image = pct_adding > config.percent_added and not skip_for_date

        if include_image:
            dates_added.add(tif_datetime)

        # Save stats for all UDMs
        if not skip_for_date:
            item_coverage.append(
                {
                    "ordered_idx": idx,
                    "asset_id": cleaned_asset_id(udm_paths[int(idx)]),
                    "clear_coverge_pct": round(clear_coverage, 4),
                    "intersection_pct": round(intersection_pct, 4),
                    "pct_adding": round(pct_adding, 4),
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
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame | None:
    grid_id = grid_path.stem
    udm_paths = tif_paths(results_grid_dir / "udm")

    if not len(udm_paths):
        logger.debug(f"No UDMs for {grid_path.stem}")
        return None

    geojson_file = results_grid_dir / "search_geometries.geojson"
    gdf = gpd.read_file(geojson_file)

    # Choose CRS to work in
    crs = find_most_common_crs(udm_paths)
    udm_gdf = gdf.to_crs(crs)

    # Load Target Grid and convert grid to udm crs
    grid = open_and_convert_grid(grid_path, crs)

    with open(results_grid_dir / "filtered_search_results.json") as f:
        grid_search_data = json.load(f)
        ground_sample_distance = grid_search_data[0]["properties"]["pixel_resolution"]

    # Create the new consistent grid which all UDMs wil be cropped to
    profile_update = create_polygon_aligned_profile_update(grid, crs, ground_sample_distance)
    target_shape = (profile_update["height"], profile_update["width"])

    # Crop the UDMs
    logger.debug("Cropping UDMs & calculating coverage")
    coverages = []
    # Remove reprojected and cropped intermediates at the end
    with tempfile.TemporaryDirectory() as tempdir:
        for udm_path in udm_paths:
            asset_id = cleaned_asset_id(udm_path)
            tif_datetime = parse_acquisition_datetime(udm_path)

            udm_data = next(d for d in grid_search_data if d.get("id", "") == asset_id)
            clear_percent = udm_data.get("properties", {}).get("clear_percent")

            item_geom: Polygon = udm_gdf[udm_gdf.id == asset_id].geometry.iloc[0]  # type: ignore
            intersection_pct = calculate_intersection_pct(grid, item_geom)

            # We COULD do this during udm query time but this allows changing the filtering level
            # after downloading based on number of results.
            if config.clear_percent and clear_percent < config.clear_percent:
                logger.debug(
                    f"Skipping {asset_id} b/c clear_percent is too low {clear_percent} < {config.clear_percent}"
                )
                coverages.append((np.zeros(target_shape, dtype=np.uint8), 0.0, intersection_pct, tif_datetime))
                continue

            # Get the UDM in the consistent grid. Do not retain the intermediates.
            temp_path = Path(tempdir) / udm_path.name
            try:
                result = reproject_and_crop_to_grid(
                    tif_path=udm_path,
                    grid_geom=grid,
                    profile_update=profile_update,
                    repro_path=temp_path,
                    out_path=None,
                    bands=[SHADOW_BAND, LIGHT_HAZE_BAND, HEAVY_HAZE_BAND, CLOUD_BAND, CONFIDENCE_BAND],
                )
            except WarpOperationError as e:
                logger.exception(e)
                log_structured_failure(
                    save_path=results_grid_dir,
                    run_id=cleaned_asset_id(udm_path),
                    category=CATEGORY,
                    payload={
                        "grid_id": grid_id,
                        "step": "reproject_and_crop_to_grid",
                        "error": str(e),
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                    },
                )
                continue

            clear_img = result[:-1].sum(axis=0) == 0
            nodata = result[-1] < 1.0
            clear_img[nodata] = 0
            clear_coverage = calculate_mask_coverage(
                clear_img,
                grid,
                ground_sample_distance,
            )
            coverages.append((clear_img, clear_coverage, intersection_pct, tif_datetime))

    # Use udms in most to least coverage order
    coverage_order = np.argsort([coverage for _, coverage, _, _ in coverages])[::-1].tolist()

    # Find UDMs which improve overall grid coverage

    # A grid of coverage counters
    coverage_count = np.zeros((profile_update["height"], profile_update["width"]), dtype=np.int32)
    # Area covered by the target grid
    grid_pixel_area = grid.area / ground_sample_distance**2

    item_coverage = update_coverage(
        coverage_order,
        coverages,
        udm_paths,
        coverage_count,
        grid_pixel_area,
        skip_same_range_days=config.skip_same_range_days,
        config=config,
    )
    included_item_idxes = {d["ordered_idx"] for d in item_coverage}

    if config.use_same_range_if_neccessary:
        skipped_coverage_order = [i for i in coverage_order if i not in included_item_idxes]
        skipped_item_coverage = update_coverage(
            skipped_coverage_order,
            coverages,
            udm_paths,
            coverage_count,
            grid_pixel_area,
            skip_same_range_days=0,
            config=config,
        )
    else:
        skipped_item_coverage = []

    return pd.DataFrame(item_coverage + skipped_item_coverage)


def udm_select(
    config_file: Path,
    start_date: datetime,
    end_date: datetime,
) -> None:
    config, save_path = create_config(config_file, start_date=start_date, end_date=end_date)

    setup_logger(save_path, log_filename="udm_select.log")

    run_id = uuid4().hex
    logger.info(
        f"Run id={run_id} Selecting best UDMs for start_date={start_date} end_date={end_date} grids={config.grid_dir} to={save_path}"
    )

    in_notebook = is_notebook()

    tqdm = get_tqdm(use_async=False, in_notebook=in_notebook)

    for grid_path in tqdm(geojson_paths(config.grid_dir, in_notebook=in_notebook, check_crs=False)):
        grid_id = grid_path.stem
        logger.debug(f"Selecting best UDMs for {grid_id}")

        results_grid_dir = save_path / grid_id
        grid_udm_dir = results_grid_dir / "udm"
        if not grid_udm_dir.exists():
            logger.debug(f"No udms for {grid_id}")
            continue

        csv_path = results_grid_dir / config.udm_select_file_name
        if csv_path.exists():
            logger.debug(f"Download list exists for {grid_id}. Skipping...")
            continue

        try:
            coverage_df = calculate_udm_coverages(
                results_grid_dir=results_grid_dir,
                grid_path=grid_path,
                config=config,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as e:
            logger.error(f"Grid {grid_id} failed in udm_select: {e}")
            log_structured_failure(
                save_path=save_path,
                run_id=run_id,
                category=CATEGORY,
                payload={
                    "grid_id": grid_id,
                    "step": "calculate_udm_coverages",
                    "error": str(e),
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                },
            )
            continue

        if coverage_df is not None:
            coverage_df.to_csv(csv_path, index=False)


@click.command()
@click.option("-c", "--config-file", type=click.Path(exists=True), required=True)
@click.option(
    "-s",
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Start date in YYYY-MM-DD format.",
    required=True,
)
@click.option(
    "-e", "--end-date", type=click.DateTime(formats=["%Y-%m-%d"]), help="End date in YYYY-MM-DD format.", required=True
)
def main(
    config_file: Path,
    start_date: datetime,
    end_date: datetime,
):
    config_file = Path(config_file)

    udm_select(config_file=config_file, start_date=start_date, end_date=end_date)

    logger.info("Done!")


if __name__ == "__main__":
    main()
