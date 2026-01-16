import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import click
import rasterio
import rasterio.errors
from dotenv import find_dotenv, load_dotenv
from planet import DataClient, Session

from src.config import DownloadConfig, udm_asset_string
from src.scripts.udm_activate import get_search_results
from src.util import (
    check_and_create_env,
    create_config,
    get_tqdm,
    is_notebook,
    log_structured_failure,
    retry_task,
    run_async_function,
    setup_logger,
)

logger = logging.getLogger(__name__)
CATEGORY = Path(__file__).stem


# Asynchronously downloads a single udm asset for the given item.
# Activates and waits for the asset to be ready before downloading it to the specified path.
# Skips files that already exist.
async def download_udm(
    order: dict,
    sess: Session,
    grid_save_path: Path,
    config: DownloadConfig,
    step_progress_bars: dict,
    sem: asyncio.Semaphore,
) -> tuple[str, str, str, datetime, Exception] | None:
    async with sem:
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
            return await cl.get_asset(
                item_type_id=order["properties"]["item_type"], item_id=udm_id, asset_type_id=asset_type_id
            )

        try:
            asset_desc = await retry_task(get_asset, config.download_retries_max, config.download_backoff)
        except Exception as e:
            step_progress_bars["get_asset"].update(1)
            return (grid_id, udm_id, "get_asset", datetime.now(), e)
        step_progress_bars["get_asset"].update(1)

        # Activate Asset
        async def activate_asset():
            await cl.activate_asset(asset=asset_desc)

        try:
            await retry_task(activate_asset, config.download_retries_max, config.download_backoff)
        except Exception as e:
            step_progress_bars["activate_asset"].update(1)
            return (grid_id, udm_id, "activate_asset", datetime.now(), e)
        step_progress_bars["activate_asset"].update(1)

        # Wait for Asset
        async def wait_asset():
            await cl.wait_asset(asset=asset_desc, delay=config.client_delay, max_attempts=config.client_max_attempts)

        try:
            _ = await retry_task(wait_asset, config.download_retries_max, config.download_backoff)
        except Exception as e:
            step_progress_bars["wait_asset"].update(1)
            return (grid_id, udm_id, "wait_asset", datetime.now(), e)
        step_progress_bars["wait_asset"].update(1)

        # Download Asset
        # NOTE: Planet download URLs/tokens can expire (often ~1–2 hours).
        # Always fetch a fresh asset descriptor immediately before downloading.
        async def download_asset():
            fresh_asset_desc = await cl.get_asset(
                item_type_id=order["properties"]["item_type"],
                item_id=udm_id,
                asset_type_id=asset_type_id,
            )

            pth = await cl.download_asset(
                asset=fresh_asset_desc,
                directory=output_path,
                overwrite=False,
                progress_bar=False,
            )

            # Ensure pth is valid, otherwise remove and (hopefully) retry
            try:
                with rasterio.open(pth) as src:
                    src.meta
            except rasterio.errors.RasterioIOError as e:
                os.remove(pth)
                raise e

        try:
            await retry_task(download_asset, config.download_retries_max, config.download_backoff)
        except Exception as e:
            step_progress_bars["download_asset"].update(1)
            return (grid_id, udm_id, "download_asset", datetime.now(), e)
        step_progress_bars["download_asset"].update(1)


# Asynchronously downloads all udm assets for the given list of items.
async def download_all_udms(
    item_lists: list[tuple[dict, Path]],
    sess: Session,
    config: DownloadConfig,
    in_notebook: bool,
    save_path: Path,
    run_id: str,
    start_date: datetime,
    end_date: datetime,
) -> None:
    logger.info(f"Downloading {len(item_lists)} udm items")

    total_assets = len(item_lists)

    # Get notebook vs async progress bar class
    tqdm = get_tqdm(use_async=True, in_notebook=in_notebook)

    # download items with limited concurrency and one progress bar
    sem = asyncio.Semaphore(config.max_concurrent_download_tasks)

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
            asyncio.create_task(download_udm(item, sess, grid_save_path, config, step_progress_bars, sem))
            for item, grid_save_path in item_lists
        ]

        # Gather all results (None if success, tuple if failure)
        results = await asyncio.gather(*tasks)

    # Filter out successful tasks and report failures
    failures = [res for res in results if res is not None]

    if failures:
        logger.error("\n[FAILED] Failed Tasks Summary:")
        for grid_id, asset_id, step, timestamp, error in failures:
            logger.error(f" - Grid {grid_id} Asset {asset_id}: Failed at {step} with error: {error}")
            log_structured_failure(
                save_path=save_path,
                run_id=run_id,
                category=CATEGORY,
                payload={
                    "grid_id": grid_id,
                    "asset_id": asset_id,
                    "step": step,
                    "error": repr(error),
                    "error_type": type(error).__name__,
                    "error_args": error.args,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "timestamp": timestamp.isoformat() + "Z",
                },
            )
    else:
        logger.info("All assets processed successfully!")


# Main loop. Download all overlapping UDMs for a given date and directory of grids.
async def main_loop(
    config: DownloadConfig,
    save_path: Path,
    in_notebook: bool,
    run_id: str,
    start_date: datetime,
    end_date: datetime,
) -> None:
    to_download = get_search_results(config, save_path, in_notebook)

    async with Session() as sess:
        # loop through and download all the UDM2 files for the given date and grid
        await download_all_udms(
            item_lists=to_download,
            sess=sess,
            config=config,
            in_notebook=in_notebook,
            save_path=save_path,
            run_id=run_id,
            start_date=start_date,
            end_date=end_date,
        )


def udm_download(
    config_file: Path,
    start_date: datetime,
    end_date: datetime,
):
    config, save_path = create_config(config_file, start_date=start_date, end_date=end_date)

    setup_logger(save_path, log_filename="udm_download.log")

    run_id = uuid4().hex
    logger.info(
        f"Run id={run_id} Downloading UDMs for start_date={start_date} end_date={end_date} grids={config.grid_dir} to={save_path}"
    )

    in_notebook = is_notebook()

    return run_async_function(main_loop(config, save_path, in_notebook, run_id, start_date, end_date))


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

    udm_download(config_file=config_file, start_date=start_date, end_date=end_date)


if __name__ == "__main__":
    main()
