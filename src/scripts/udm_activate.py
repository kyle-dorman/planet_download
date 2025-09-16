import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Sequence

import click
from dotenv import find_dotenv, load_dotenv
from planet import DataClient, Session

from src.config import DownloadConfig, udm_asset_string
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


# Asynchronously activated a single udm asset for the given item.
# Activates and waits for the asset to be ready.
# Skips files that already exist.
async def activate_region_udm(
    order: dict,
    sess: Session,
    grid_save_path: Path,
    config: DownloadConfig,
    step_progress_bars: dict,
    sem: asyncio.Semaphore,
) -> tuple[str, str, str, str] | None:
    udm_id = order["id"]
    grid_id = grid_save_path.name

    output_path = grid_save_path / "udm"
    output_path.mkdir(parents=True, exist_ok=True)

    # Exit early if there is a file that matches the order_id.
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


# Asynchronously activated all udm assets for the given list of items.
async def activate_all_udms(
    item_lists: list[tuple[dict, Path]], sess: Session, config: DownloadConfig, in_notebook: bool
) -> None:
    logger.info(f"Activating {len(item_lists)} udm items")

    total_assets = len(item_lists)

    # Get notebook vs async progress bar class
    tqdm = get_tqdm(use_async=True, in_notebook=in_notebook)

    # activate items with limited concurrency and one progress bar
    sem = asyncio.Semaphore(config.max_concurrent_tasks)

    # Initialize progress bars for each step
    with (
        tqdm(total=total_assets, desc="Step 1: Getting Assets", position=0, dynamic_ncols=True) as get_pbar,
        tqdm(total=total_assets, desc="Step 2: Activating Assets", position=1, dynamic_ncols=True) as activate_pbar,
    ):

        # Dictionary to track progress of each step
        step_progress_bars = {
            "get_asset": get_pbar,
            "activate_asset": activate_pbar,
        }

        # Run all tasks and collect results
        tasks = [
            asyncio.create_task(activate_region_udm(item, sess, grid_save_path, config, step_progress_bars, sem))
            for item, grid_save_path in item_lists
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


def get_grid_search_results(
    save_path: Path,
    grid_path: Path,
) -> Sequence[tuple[dict, Path]]:
    """Handles one grid: Loads search results and returns tuples of save path and search result."""
    grid_id = grid_path.stem
    grid_save_path = save_path / grid_id
    if grid_save_path.exists() and (grid_save_path / "images_to_activate.csv").exists():
        logger.debug(f"Already filtered grid {grid_id}")
        return []
    results_path = grid_save_path / "filtered_search_results.json"
    assert results_path.exists(), f"Missing file {results_path}"

    with open(results_path) as f:
        filtered_item_list = json.load(f)

    return [(item, grid_save_path) for item in filtered_item_list]


# Gets a list of all UDMs which need to be activated across all grids.
def get_search_results(config: DownloadConfig, save_path: Path, in_notebook: bool) -> list[tuple[dict, Path]]:
    grid_paths = geojson_paths(config.grid_dir, in_notebook=in_notebook, check_crs=False)

    to_activate = []
    for grid_path in grid_paths:
        to_activate.extend(get_grid_search_results(save_path, grid_path))

    return to_activate


# Main loop. Download all overlapping UDMs for a given date and directory of grids.
async def main_loop(config: DownloadConfig, save_path: Path, in_notebook: bool) -> None:
    to_activate = get_search_results(config, save_path, in_notebook)

    async with Session() as sess:
        # loop through and activate all the UDM2 files for the given date and grid
        await activate_all_udms(to_activate, sess, config, in_notebook)


def udm_activate(
    config_file: Path,
    start_date: datetime,
    end_date: datetime,
):
    config, save_path = create_config(config_file, start_date=start_date, end_date=end_date)

    setup_logger(save_path, log_filename="udm_activate.log")

    logger.info(
        f"Activating UDMs for start_date={start_date} end_date={end_date} grids={config.grid_dir} to={save_path}"
    )

    in_notebook = is_notebook()

    return run_async_function(main_loop(config, save_path, in_notebook))


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

    udm_activate(config_file=config_file, start_date=start_date, end_date=end_date)


if __name__ == "__main__":
    main()
