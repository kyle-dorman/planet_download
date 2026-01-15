import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

import click
from dotenv import find_dotenv, load_dotenv
from planet import DataClient, Session, data_filter

from src.config import DownloadConfig, ItemType, QualityCategory, planet_asset_string
from src.grid import grids_overlap, load_grid
from src.util import (
    check_and_create_env,
    create_config,
    geojson_paths,
    get_tqdm,
    has_crs,
    is_notebook,
    log_structured_failure,
    retry_task,
    run_async_function,
    setup_logger,
)

logger = logging.getLogger(__name__)

CATEGORY = Path(__file__).stem


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

    with save_path.open("w") as f:
        json.dump(geoms, f)

    logger.debug(f"Saved search geometry to {save_path}")


# Define the search filters used to find the UDMs
def create_search_filter(start_date: datetime, end_date: datetime, config: DownloadConfig) -> dict:
    # date range filter
    date_range_filter = data_filter.date_range_filter("acquired", gte=start_date, lt=end_date)

    # Asset filter
    asset_type = planet_asset_string(config)
    asset_filter = data_filter.asset_filter([asset_type])

    # Minimum "clear" pct of the image.
    clear_percent_filter = data_filter.range_filter("clear_percent", gte=config.clear_percent)

    # Only get data we can download
    permission_filter = data_filter.permission_filter()

    filters = [
        date_range_filter,
        asset_filter,
        clear_percent_filter,
        permission_filter,
    ]

    # Only get "quality" images
    if config.quality_category is not None and config.quality_category == QualityCategory.Standard:
        quality_category_filter = data_filter.std_quality_filter()
        filters.append(quality_category_filter)

    # Define the filter for Dove/SuperDove instruments
    if config.instrument is not None and len(config.instrument):
        instrument_filter = data_filter.string_in_filter("instrument", [inst.value for inst in config.instrument])
        filters.append(instrument_filter)

    # Set publishing level filter
    if config.item_type == ItemType.PSScene:
        if config.publishing_stage is not None:
            publishing_filter = data_filter.string_in_filter("publishing_stage", [config.publishing_stage.value])
            filters.append(publishing_filter)

        # Has ground control points
        ground_control_filter = data_filter.string_in_filter("ground_control", [str(config.ground_control).lower()])
        filters.append(ground_control_filter)

    # combine all of the filters
    all_filters = data_filter.and_filter(filters)

    return all_filters


# Asynchronously creates a search request with the given search filter. Returns the created search request.
async def create_search(
    sess: Session, search_name: str, search_filter: dict, grid_geojson: dict, config: DownloadConfig
) -> dict:
    logger.debug(f"Creating search request {search_name}")

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


# Asynchronously performs a search using the given search request.
# Returns a list of items found by the search.
def do_search(sess: Session, search_request: dict, config: DownloadConfig) -> AsyncIterator[dict]:
    logger.debug(f"Search for udm2 matches {search_request['id']}")

    items = DataClient(sess).run_search(search_id=search_request["id"], limit=config.udm_limit)

    logger.debug(f"Executed udm2 search {search_request['id']}")

    return items


# Asynchronously performs a search for imagery using the given geometry, and start date.
# Returns a list of items found by the search.
async def search(
    sess: Session,
    grid_path: Path,
    grid_save_path: Path,
    config: DownloadConfig,
    start_date: datetime,
    end_date: datetime,
) -> AsyncIterator[dict]:
    search_request_path = grid_save_path / "search_request.json"
    if search_request_path.exists():
        with search_request_path.open(encoding="utf-8") as f:
            search_request = json.load(f)
    else:
        with open(grid_path) as file:
            grid_geojson = json.load(file)

        search_filter = create_search_filter(start_date, end_date, config)
        search_name = "pldl_" + str(uuid4())

        search_request = await create_search(
            sess=sess, search_name=search_name, search_filter=search_filter, grid_geojson=grid_geojson, config=config
        )

        with search_request_path.open("w") as f:
            json.dump(search_request, f)

    # search for matches
    return do_search(sess, search_request, config)


async def run_search(
    sess: Session,
    config: DownloadConfig,
    save_path: Path,
    start_date: datetime,
    end_date: datetime,
    grid_path: Path,
    pbar,
    sem: asyncio.Semaphore,
    run_id: str,
) -> None:
    """Handles one grid: searches, filters, and returns search result tuples."""
    grid_id = grid_path.stem
    grid_save_path = save_path / grid_id
    if grid_save_path.exists() and (grid_save_path / config.udm_select_file_name).exists():
        logger.debug(f"Already filtered grid {grid_id}")
        return

    results_path = grid_save_path / "filtered_search_results.json"
    if results_path.exists():
        return

    has_crs(grid_path)
    grid_poly = load_grid(grid_path)
    grid_save_path.mkdir(parents=True, exist_ok=True)

    async with sem:
        try:

            async def _collect_lazy():
                lazy = await search(
                    sess=sess,
                    grid_path=grid_path,
                    grid_save_path=grid_save_path,
                    config=config,
                    start_date=start_date,
                    end_date=end_date,
                )
                return [i async for i in lazy]

            item_list = await retry_task(_collect_lazy, config.download_retries_max, config.download_backoff)

            filtered_item_list = []
            for item in item_list:
                if grids_overlap(item, grid_poly, config.percent_added):
                    filtered_item_list.append(item)

            with results_path.open("w") as f:
                json.dump(filtered_item_list, f)
            if len(filtered_item_list):
                save_search_geom(filtered_item_list, grid_save_path / "search_geometries.geojson")

        except Exception as error:
            logger.error(f"Grid {grid_id} failed: {error}")
            log_structured_failure(
                save_path=save_path,
                run_id=run_id,
                category=CATEGORY,
                payload={
                    "grid_id": grid_id,
                    "error": repr(error),
                    "error_type": type(error).__name__,
                    "error_args": error.args,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                },
            )

    pbar.update(1)


# Saves a list of all UDMs which need to be downloaded across all grids.
async def run_searches_gather(
    sess: Session,
    config: DownloadConfig,
    save_path: Path,
    start_date: datetime,
    end_date: datetime,
    in_notebook: bool,
    run_id: str,
) -> None:
    grid_paths = geojson_paths(config.grid_dir, in_notebook=in_notebook, check_crs=False)
    total = len(grid_paths)

    # search items with limited concurrency and one progress bar
    sem = asyncio.Semaphore(config.max_concurrent_tasks)

    tqdm = get_tqdm(use_async=True, in_notebook=in_notebook)
    with tqdm(total=total, desc="Create Search", position=0, dynamic_ncols=True) as pbar:
        coros = [
            run_search(sess, config, save_path, start_date, end_date, grid_path, pbar, sem, run_id)
            for grid_path in grid_paths
        ]
        await asyncio.gather(*coros)


# Main loop. Save all overlapping UDMs for a given date and directory of grids.
async def main_loop(
    config: DownloadConfig,
    save_path: Path,
    start_date: datetime,
    end_date: datetime,
    in_notebook: bool,
    run_id: str,
) -> None:
    async with Session() as sess:
        await run_searches_gather(sess, config, save_path, start_date, end_date, in_notebook, run_id)


def udm_search(
    config_file: Path,
    start_date: datetime,
    end_date: datetime,
):
    config, save_path = create_config(config_file, start_date=start_date, end_date=end_date)

    setup_logger(save_path, log_filename="udm_search.log")

    run_id = uuid4().hex
    logger.info(
        f"Run id={run_id} Searching for UDMs for start_date={start_date} end_date={end_date} "
        f"grids={config.grid_dir} to={save_path}"
    )

    in_notebook = is_notebook()

    return run_async_function(main_loop(config, save_path, start_date, end_date, in_notebook, run_id))


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

    udm_search(config_file=config_file, start_date=start_date, end_date=end_date)


if __name__ == "__main__":
    main()
