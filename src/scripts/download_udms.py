import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

import click
from dateutil.relativedelta import relativedelta
from dotenv import find_dotenv, load_dotenv
from planet import DataClient, Session, data_filter
from shapely.geometry import Polygon, shape

from src.config import DownloadConfig
from src.grid import calculate_intersection_pct
from src.util import (
    check_and_create_env,
    create_config,
    geojson_paths,
    get_tqdm,
    is_notebook,
    retry_task,
    run_async_function,
    setup_logger,
)

logger = logging.getLogger(__name__)


# Asynchronously creates a search request with the given search filter. Returns the created search request.
async def create_search(sess: Session, search_name: str, search_filter: dict, config: DownloadConfig) -> dict:
    logger.debug(f"Creating search request {search_name}")

    search_request = await DataClient(sess).create_search(
        name=search_name,
        search_filter=search_filter,
        item_types=[config.item_type],
    )

    logger.debug(f"Created search request {search_name} {search_request['id']}")

    return search_request


# Asynchronously performs a search using the given search request.
# Returns a list of items found by the search.
async def do_search(sess: Session, search_request: dict, config: DownloadConfig) -> AsyncIterator[dict]:
    logger.debug(f"Search for udm2 matches {search_request['id']}")

    items = DataClient(sess).run_search(search_id=search_request["id"], limit=config.udm_limit)

    logger.debug(f"Executed udm2 search {search_request['id']}")

    return items


# Define the search filters used to find the UDMs
def create_search_filter(grid_path: Path, min_acquired: datetime, config: DownloadConfig) -> dict:
    with open(grid_path) as file:
        grid_geojson = json.load(file)

    # geometry filter
    geom_filter = data_filter.geometry_filter(grid_geojson)

    # date range filter
    date_range_filter = data_filter.date_range_filter(
        "acquired", gte=min_acquired, lt=min_acquired + relativedelta(months=1)
    )

    # Asset filter
    asset_type = config.asset_type.planet_asset_string(min_acquired)
    superdove_filter = data_filter.asset_filter([asset_type])

    # Has ground control points
    ground_control_filter = data_filter.string_in_filter("ground_control", [str(config.ground_control).lower()])

    # Minimum "clear" pct of the image.
    clear_percent_filter = data_filter.range_filter("clear_percent", gte=config.clear_percent)

    # Set publishing level filter
    publishing_filter = data_filter.string_in_filter("publishing_stage", [config.publishing_stage])

    # combine all of the filters
    return data_filter.and_filter(
        [
            geom_filter,
            date_range_filter,
            superdove_filter,
            ground_control_filter,
            clear_percent_filter,
            publishing_filter,
        ]
    )


# Asynchronously performs a search for imagery using the given geometry, and start date.
# Returns a list of itemsfound by the search.
async def search(
    sess: Session, grid_path: Path, config: DownloadConfig, save_path: Path, min_acquired: datetime
) -> AsyncIterator[dict]:
    search_filter = create_search_filter(grid_path, min_acquired, config)
    with open(save_path / "search_filter.json", "w") as f:
        json.dump(search_filter, f)

    search_name = f"{config.udm_search_name}_{config.grid_dir.stem}_{min_acquired.year}_{min_acquired.month}"
    search_request = await create_search(sess, search_name, search_filter, config)
    with open(save_path / "search_request.json", "w") as f:
        json.dump(search_request, f)

    # search for matches
    return await do_search(sess, search_request, config)


# Saves the geometries of the search results to a GeoJSON file in the specified output path.
def save_search_geom(item_list: list[dict], save_path: Path) -> None:
    geoms = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": item["id"], "pixel_resolution": item["properties"]["pixel_resolution"]},
                "geometry": item["geometry"],
            }
            for item in item_list
        ],
    }

    with open(save_path, "w") as f:
        json.dump(geoms, f)

    logger.debug(f"Saved search geometry to {save_path}")


# Filters UDMs that intersect with the grid less than the DownloadConfig.percent_added.
# You can do this in the Planet web tool but not the API.
def filter_grid_intersection(grid_path: Path, item_list: list[dict], config: DownloadConfig) -> list[dict]:
    with open(grid_path) as f:
        grid_geojson = json.load(f)
        grid_geom: Polygon = shape(grid_geojson["features"][0]["geometry"])  # type: ignore

    out = []
    for item in item_list:
        item_grid: Polygon = shape(item["geometry"])  # type: ignore
        pct_intersection = calculate_intersection_pct(grid_geom, item_grid)

        if pct_intersection < config.percent_added:
            continue

        out.append(item)

    return out


# Asynchronously downloads a single udm asset for the given item.
# Activates and waits for the asset to be ready before downloading it to the specified path.
# Skips files that already exist.
async def download_udm(
    order: dict, sess: Session, output_path: Path, config: DownloadConfig, step_progress_bars: dict
) -> tuple[str, str, str, str] | None:
    udm_id = order["id"]
    grid_id = output_path.parent.name

    # Exit early if there is a directory that matches the order_id.
    if any(pth.stem.startswith(udm_id) for pth in output_path.iterdir()):
        for v in step_progress_bars.values():
            v.update(1)
        return

    cl = DataClient(sess)

    # Get Asset
    async def get_asset():
        return await cl.get_asset(
            item_type_id=order["properties"]["item_type"], item_id=udm_id, asset_type_id=config.udm_asset_type
        )

    try:
        asset_desc = await retry_task(get_asset, config.download_retries_max, config.download_backoff)
    except Exception as e:
        step_progress_bars["get_asset"].update(1)
        return (grid_id, udm_id, "get_asset", str(e))
    step_progress_bars["get_asset"].update(1)

    # Activate Asset
    async def activate_asset():
        await cl.activate_asset(asset=asset_desc)

    try:
        await retry_task(activate_asset, config.download_retries_max, config.download_backoff)
    except Exception as e:
        step_progress_bars["activate_asset"].update(1)
        return (grid_id, udm_id, "activate_asset", str(e))
    step_progress_bars["activate_asset"].update(1)

    # Wait for Asset
    async def wait_asset():
        return await cl.wait_asset(asset=asset_desc)

    try:
        asset = await retry_task(wait_asset, config.download_retries_max, config.download_backoff)
    except Exception as e:
        step_progress_bars["wait_asset"].update(1)
        return (grid_id, udm_id, "wait_asset", str(e))
    step_progress_bars["wait_asset"].update(1)

    # Download Asset
    async def download_asset():
        await cl.download_asset(asset=asset, directory=output_path, overwrite=False, progress_bar=False)

    try:
        await retry_task(download_asset, config.download_retries_max, config.download_backoff)
    except Exception as e:
        step_progress_bars["download_asset"].update(1)
        return (grid_id, udm_id, "download_asset", str(e))
    step_progress_bars["download_asset"].update(1)


# Asynchronously downloads all udm assets for the given list of items.
async def download_all_udms(
    item_lists: list[tuple[dict, Path]], sess: Session, config: DownloadConfig, in_notebook: bool
) -> None:
    logger.info(f"Downloading {len(item_lists)} udm items")

    total_assets = len(item_lists)

    # Get notebook vs async progress bar class
    tqdm = get_tqdm(use_async=True, in_notebook=in_notebook)

    # Initialize progress bars for each step
    with (
        tqdm(total=total_assets, desc="Step 1: Getting Assets", position=0, dynamic_ncols=True) as get_pbar,
        tqdm(total=total_assets, desc="Step 2: Activating Assets", position=1, dynamic_ncols=True) as activate_pbar,
        tqdm(total=total_assets, desc="Step 3: Waiting for Assets", position=2, dynamic_ncols=True) as wait_pbar,
        tqdm(total=total_assets, desc="Step 4: Downloading Assets", position=3, dynamic_ncols=True) as download_pbar,
    ):

        # Dictionary to track progress of each step
        step_progress_bars = {
            "get_asset": get_pbar,
            "activate_asset": activate_pbar,
            "wait_asset": wait_pbar,
            "download_asset": download_pbar,
        }

        # Run all tasks and collect results
        tasks = [
            asyncio.create_task(download_udm(item, sess, output_path, config, step_progress_bars))
            for item, output_path in item_lists
        ]

        # Gather all results (None if success, tuple if failure)
        results = await asyncio.gather(*tasks)

    # Filter out successful tasks and report failures
    failures = [res for res in results if res is not None]

    if failures:
        logger.error("\n❌ Failed Tasks Summary:")
        for grid_id, asset_id, step, error in failures:
            logger.error(f" - Grid {grid_id} Asset {asset_id}: Failed at {step} with error: {error}")
    else:
        logger.info("All assets processed successfully!")


# Gets a list of all UDMs which need to be downloaded across all grids.
async def get_download_list(
    sess: Session, config: DownloadConfig, save_path: Path, imagery_date: datetime
) -> list[tuple[dict, Path]]:
    to_download = []
    for grid_path in geojson_paths(config.grid_dir):
        assert grid_path.suffix == ".geojson", f"Invalid path, {grid_path.name}"

        grid_save_path = save_path / grid_path.stem
        grid_save_path.parent.mkdir(exist_ok=True, parents=True)
        udm_save_dir = grid_save_path / "udm"
        udm_save_dir.mkdir(exist_ok=True, parents=True)

        search_results_path = grid_save_path / "search_results.json"
        if search_results_path.exists():
            with open(search_results_path) as f:
                item_list = json.load(f)
        else:
            # define the original item list. This is all imagery for the given date and grid
            lazy_item_list = await search(sess, grid_path, config, save_path, imagery_date)
            item_list = [i async for i in lazy_item_list]

            item_list = filter_grid_intersection(grid_path, item_list, config)
            logger.info(f"Found {len(item_list)} matching grids for {grid_path.stem}")

            # Save the results
            with open(search_results_path, "w") as f:
                json.dump(item_list, f)
            save_search_geom(item_list, grid_save_path / "search_geometries.geojson")

        if len(item_list) == 0:
            logger.warning(f"No matches for {grid_path.stem} {imagery_date}")
            continue

        for item in item_list:
            to_download.append((item, udm_save_dir))

    return to_download


# Main loop. Download all overlapping UDMs for a given date and directory of grids.
async def main_loop(config: DownloadConfig, save_path: Path, imagery_date: datetime, in_notebook: bool) -> None:
    async with Session() as sess:
        to_download = await get_download_list(sess, config, save_path, imagery_date)

        # loop through and download all the UDM2 files for the given date and grid
        await download_all_udms(to_download, sess, config, in_notebook)


def download_udms(
    config_file: Path,
    year: int,
    month: int,
):
    config, save_path = create_config(config_file, year=year, month=month)

    setup_logger(save_path, log_filename="download_udms.log")

    imagery_date = datetime(year, month, 1)

    logger.info(f"Downloading UDMs for year={year} month={month} grids={config.grid_dir} to={save_path}")

    in_notebook = is_notebook()

    return run_async_function(main_loop(config, save_path, imagery_date, in_notebook))


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

    # Set the PlanetAPI Key in .env file if not set
    check_and_create_env()

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv(raise_error_if_not_found=True))

    download_udms(config_file=config_file, month=month, year=year)


if __name__ == "__main__":
    main()
