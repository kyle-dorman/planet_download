import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Sequence

import click
from dotenv import find_dotenv, load_dotenv
from planet import DataClient, Session, data_filter

from src.config import DownloadConfig, ItemType, planet_asset_string, udm_asset_string
from src.grid import grids_overlap, load_grid
from src.util import (
    check_and_create_env,
    create_config,
    geojson_paths,
    get_tqdm,
    has_crs,
    is_notebook,
    retry_task,
    run_async_function,
    setup_logger,
)

logger = logging.getLogger(__name__)


# Asynchronously creates a search request with the given search filter. Returns the created search request.
async def create_search(
    sess: Session, search_name: str, search_filter: dict, grid_path: Path, config: DownloadConfig
) -> dict:
    logger.debug(f"Creating search request {search_name}")

    with open(grid_path) as file:
        grid_geojson = json.load(file)

    async def create_search_inner():
        return await DataClient(sess).create_search(
            name=search_name,
            search_filter=search_filter,
            item_types=[config.item_type.value],
            geometry=grid_geojson,
        )

    search_request = await retry_task(create_search_inner, config.download_retries_max, config.download_backoff)

    logger.debug(f"Created search request {search_name} {search_request['id']}")

    return search_request


# Asynchronously performs a search with the given search filter.
# Returns a list of items found by the search.
def do_search(
    sess: Session, grid_id: str, search_filter: dict, grid_geojson: dict, config: DownloadConfig
) -> AsyncIterator[dict]:
    logger.debug(f"Search for udm2 matches for grid_id {grid_id}")

    items = DataClient(sess).search(
        item_types=[config.item_type.value],
        search_filter=search_filter,
        geometry=grid_geojson,
        limit=config.udm_limit,
    )

    logger.debug(f"Executed udm2 search for grid_id {grid_id}")

    return items


# Define the search filters used to find the UDMs
def create_search_filter(start_date: datetime, end_date: datetime, grid_geojson: dict, config: DownloadConfig) -> dict:
    # geometry filter
    geom_filter = data_filter.geometry_filter(grid_geojson)

    # date range filter
    date_range_filter = data_filter.date_range_filter("acquired", gte=start_date, lt=end_date)

    # Asset filter
    asset_type = planet_asset_string(config)
    asset_filter = data_filter.asset_filter([asset_type])

    # Only get "quality" images
    quality_category_filter = data_filter.string_in_filter("quality_category", [config.quality_category])

    # Minimum "clear" pct of the image.
    clear_percent_filter = data_filter.range_filter("clear_percent", gte=config.clear_percent)

    # Only get data we can download
    permission_filter = data_filter.permission_filter()

    # Set item type
    item_filter = data_filter.string_in_filter("item_type", [config.item_type.value])

    filters = [
        geom_filter,
        date_range_filter,
        asset_filter,
        quality_category_filter,
        clear_percent_filter,
        permission_filter,
        item_filter,
    ]

    # Set publishing level filter
    if config.item_type == ItemType.PSScene:
        publishing_filter = data_filter.string_in_filter("publishing_stage", [config.publishing_stage])
        filters.append(publishing_filter)

        # Has ground control points
        ground_control_filter = data_filter.string_in_filter("ground_control", [str(config.ground_control).lower()])
        filters.append(ground_control_filter)

    # combine all of the filters
    all_filters = data_filter.and_filter(filters)

    return all_filters


# Asynchronously performs a search for imagery using the given geometry, and start date.
# Returns a list of items found by the search.
async def search(
    sess: Session,
    grid_path: Path,
    config: DownloadConfig,
    start_date: datetime,
    end_date: datetime,
) -> AsyncIterator[dict]:
    with open(grid_path) as file:
        grid_geojson = json.load(file)

    search_filter = create_search_filter(start_date, end_date, grid_geojson, config)

    return do_search(
        sess=sess, grid_id=grid_path.stem, search_filter=search_filter, grid_geojson=grid_geojson, config=config
    )


# Saves the geometries of the search results to a GeoJSON file in the specified output path.
def save_search_geom(item_list: list[dict], save_path: Path) -> None:
    geoms = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": item["id"],
                    "pixel_resolution": item["properties"]["pixel_resolution"],
                    "acquired": item["properties"]["acquired"],
                },
                "geometry": item["geometry"],
            }
            for item in item_list
        ],
    }

    with open(save_path, "w") as f:
        json.dump(geoms, f)

    logger.debug(f"Saved search geometry to {save_path}")


# Asynchronously downloads a single udm asset for the given item.
# Activates and waits for the asset to be ready before downloading it to the specified path.
# Skips files that already exist.
async def download_udm(
    order: dict,
    sess: Session,
    output_path: Path,
    config: DownloadConfig,
    step_progress_bars: dict,
    sem: asyncio.Semaphore,
) -> tuple[str, str, str, str] | None:
    udm_id = order["id"]
    grid_id = output_path.parent.name

    # Exit early if there is a directory that matches the order_id.
    if any(pth.stem.startswith(udm_id) for pth in output_path.iterdir()):
        for v in step_progress_bars.values():
            v.update(1)
        return

    cl = DataClient(sess)

    asset_type_id = udm_asset_string(config)

    # Get Asset
    async def get_asset():
        async with sem:
            return await cl.get_asset(
                item_type_id=order["properties"]["item_type"], item_id=udm_id, asset_type_id=asset_type_id
            )

    try:
        asset_desc = await retry_task(get_asset, config.download_retries_max, config.download_backoff)
    except Exception as e:
        step_progress_bars["get_asset"].update(1)
        return (grid_id, udm_id, "get_asset", str(e))
    step_progress_bars["get_asset"].update(1)

    # Activate Asset
    async def activate_asset():
        async with sem:
            await cl.activate_asset(asset=asset_desc)

    try:
        await retry_task(activate_asset, config.download_retries_max, config.download_backoff)
    except Exception as e:
        step_progress_bars["activate_asset"].update(1)
        return (grid_id, udm_id, "activate_asset", str(e))
    step_progress_bars["activate_asset"].update(1)

    # Wait for Asset
    async def wait_asset():
        async with sem:
            return await cl.wait_asset(asset=asset_desc)

    try:
        asset = await retry_task(wait_asset, config.download_retries_max, config.download_backoff)
    except Exception as e:
        step_progress_bars["wait_asset"].update(1)
        return (grid_id, udm_id, "wait_asset", str(e))
    step_progress_bars["wait_asset"].update(1)

    # Download Asset
    async def download_asset():
        async with sem:
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

    # download items with limited concurrency and one progress bar
    sem = asyncio.Semaphore(config.max_concurrent_tasks)

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
            asyncio.create_task(download_udm(item, sess, output_path, config, step_progress_bars, sem))
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


async def process_grid(
    sess: Session,
    config: DownloadConfig,
    save_path: Path,
    start_date: datetime,
    end_date: datetime,
    grid_path: Path,
    pbar,
    sem: asyncio.Semaphore,
) -> Sequence[tuple[dict, Path]]:
    """Handles one grid: searches, filters, and returns download tuples."""
    to_download = []
    grid_id = grid_path.stem
    grid_save_path = save_path / grid_id
    udm_save_dir = grid_save_path / "udm"
    if grid_save_path.exists() and (grid_save_path / "images_to_download.csv").exists():
        logger.debug(f"Already filtered grid {grid_id}")
        return []
    try:
        results_path = grid_save_path / "filtered_search_results.json"
        if results_path.exists():
            with open(results_path) as f:
                filtered_item_list = json.load(f)
        else:
            has_crs(grid_path)
            grid_poly = load_grid(grid_path)

            async def _collect_lazy():
                lazy = await search(sess, grid_path, config, start_date, end_date)
                return [i async for i in lazy]

            async with sem:
                item_list = await retry_task(_collect_lazy, config.download_retries_max, config.download_backoff)

            filtered_item_list = []
            for item in item_list:
                if grids_overlap(item, grid_poly, config.percent_added):
                    filtered_item_list.append(item)

            if not len(filtered_item_list):
                logger.debug(f"No matches for {grid_path.stem}")
            else:
                udm_save_dir.mkdir(parents=True, exist_ok=True)

                with open(results_path, "w") as f:
                    json.dump(filtered_item_list, f)

                save_search_geom(filtered_item_list, grid_save_path / "search_geometries.geojson")

        to_download = [(item, udm_save_dir) for item in filtered_item_list]
    except Exception as e:
        logger.error(f"Grid {grid_path.stem} failed: {e}")

    pbar.update(1)

    return to_download


# Gets a list of all UDMs which need to be downloaded across all grids.
async def get_download_list_gather(
    sess: Session, config: DownloadConfig, save_path: Path, start_date: datetime, end_date: datetime, in_notebook: bool
) -> list[tuple[dict, Path]]:
    grid_paths = geojson_paths(config.grid_dir, in_notebook=in_notebook, check_crs=False)
    total = len(grid_paths)

    # download items with limited concurrency and one progress bar
    sem = asyncio.Semaphore(config.max_concurrent_tasks)

    tqdm = get_tqdm(use_async=True, in_notebook=in_notebook)
    with tqdm(total=total, desc="Grids", position=0, dynamic_ncols=True) as pbar:
        coros = [
            process_grid(sess, config, save_path, start_date, end_date, grid_path, pbar, sem)
            for grid_path in grid_paths
        ]
        to_downloads = await asyncio.gather(*coros)
    return [a for b in to_downloads for a in b]


# Main loop. Download all overlapping UDMs for a given date and directory of grids.
async def main_loop(
    config: DownloadConfig, save_path: Path, start_date: datetime, end_date: datetime, in_notebook: bool
) -> None:
    async with Session() as sess:
        to_download = await get_download_list_gather(sess, config, save_path, start_date, end_date, in_notebook)

        # loop through and download all the UDM2 files for the given date and grid
        await download_all_udms(to_download, sess, config, in_notebook)


def download_udms(
    config_file: Path,
    start_date: datetime,
    end_date: datetime,
):
    config, save_path = create_config(config_file, start_date=start_date, end_date=end_date)

    setup_logger(save_path, log_filename="download_udms.log")

    logger.info(
        f"Downloading UDMs for start_date={start_date} end_date={end_date} grids={config.grid_dir} to={save_path}"
    )

    in_notebook = is_notebook()

    return run_async_function(main_loop(config, save_path, start_date, end_date, in_notebook))


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

    # Set the PlanetAPI Key in .env file if not set
    check_and_create_env()

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv(raise_error_if_not_found=True))

    download_udms(config_file=config_file, start_date=start_date, end_date=end_date)


if __name__ == "__main__":
    main()
