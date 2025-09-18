import asyncio
import json
import logging
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv
from planet import OrdersClient, Session

from src.config import DownloadConfig
from src.scripts.order_create import get_order_jsons
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


def orders_to_download(results_grid_dir: Path) -> list[tuple[int, Path]]:
    order_paths = []
    for order_path in get_order_jsons(results_grid_dir):
        order_idx = int(order_path.stem.split("_")[1])
        data = (order_idx, order_path)

        manifest_path = results_grid_dir / f"manifest_{order_idx}.json"

        # If there is no "files" directory then the data wasn't downloaded
        order_files_dir = results_grid_dir / "files"
        if not order_files_dir.exists():
            order_paths.append(data)
            continue

        # If we did not recored the list of file names then the data wasn't extracted successfully.
        if not manifest_path.exists():
            order_paths.append(data)
            continue

        with open(manifest_path) as f:
            manifest = json.load(f).get("files", [])
            file_names = {Path(m["path"]).name for m in manifest}

        # Verify the data was downloaded, extracted, and matches the list of files expected.
        downloaded_files = {p.name for p in order_files_dir.iterdir()}
        missing_files = file_names - downloaded_files
        if len(missing_files):
            order_paths.append(data)
    return order_paths


def unzip_download(order_idx: int, order_request: dict, results_grid_dir: Path) -> None:
    """Unzip the downloaded zip file.

    Args:
        order_idx (int): The chunked order index
        order_request (dict): The order data
        results_grid_dir (Path): The place where per date/grid downloaded data is saved.
    """
    # Location of unziped files
    order_files_dir = results_grid_dir / "files"

    # Get the order_id to know the name of the zip file
    order_id = str(order_request["id"])

    order_download_dir = results_grid_dir / order_id
    if not order_download_dir.exists():
        logger.debug(f"Missing order download for {results_grid_dir.stem} {order_id}")
        return

    # There should just be one zip file in the download folder.
    order_download_paths = list(order_download_dir.glob("*.zip"))
    assert len(order_download_paths) == 1, order_download_paths
    order_download_path = order_download_paths[0]
    try:
        # Open the zip file and extract its contents
        with zipfile.ZipFile(order_download_path) as zip_ref:
            zip_ref.extractall(results_grid_dir)
    except zipfile.BadZipFile as e:
        logger.error(f"Path: {order_download_path}")
        raise e

    assert order_files_dir.exists(), order_files_dir

    # Move the manifest to be a per order_idx manifest
    manifest_path = results_grid_dir / "manifest.json"
    assert manifest_path.exists(), manifest_path
    shutil.move(manifest_path, results_grid_dir / f"manifest_{order_idx}.json")


def cleanup(order_request: dict, results_grid_dir: Path) -> None:
    """Remove large intermediate zip file

    Args:
        order_request (dict): The order data
        results_grid_dir (Path): The place where per date/grid downloaded data is saved.
    """
    order_id = str(order_request["id"])

    order_download_dir = results_grid_dir / order_id
    if not order_download_dir.exists():
        logger.debug(f"Missing order download for {results_grid_dir.stem} {order_id}")
        return

    shutil.rmtree(order_download_dir)


# Download an order zip file. If order folder already exists, skip over it.
async def download_single_order(
    sess: Session,
    order: dict,
    order_idx: int,
    save_dir: Path,
    config: DownloadConfig,
    step_progress_bars: dict,
    sem: asyncio.Semaphore,
) -> tuple[str, str, str] | None:
    grid_id = save_dir.stem
    order_id = str(order["id"])

    cl = OrdersClient(sess)

    # Wait for the order to be ready
    async def wait_order():
        async with sem:
            await cl.wait(order_id, delay=config.client_delay, max_attempts=config.client_max_attempts)

    try:
        await retry_task(wait_order, config.download_retries_max, config.download_backoff)
    except Exception as e:
        return (grid_id, "wait_order", str(e))
    step_progress_bars["wait_order"].update(1)

    # Download the files
    async def download_order():
        async with sem:
            await cl.download_order(order_id, directory=save_dir, overwrite=False, progress_bar=False)
            for pth in save_dir.glob("*.zip"):
                if not zipfile.is_zipfile(pth):
                    os.remove(pth)
                    raise zipfile.BadZipFile(f"File is not a zip file {pth}")

    try:
        await retry_task(download_order, config.download_retries_max, config.download_backoff)
    except Exception as e:
        return (grid_id, "download_order", str(e))
    step_progress_bars["download_order"].update(1)

    try:
        unzip_download(order_idx, order, save_dir)
    except Exception as e:
        return (grid_id, "unzip", str(e))
    step_progress_bars["unzip"].update(1)

    if config.cleanup:
        try:
            cleanup(order, save_dir)
        except Exception as e:
            return (grid_id, "cleanup", str(e))
    step_progress_bars["cleanup"].update(1)


# Download an order and retry failed downloads a fixed number of times.
async def download_orders(
    sess: Session, orders: list[tuple[int, dict, Path]], config: DownloadConfig, in_notebook: bool
) -> None:
    # Skip orders that were previously downloaded
    total_assets = len(orders)
    logger.info(f"Downloading {total_assets} orders")

    # download items with limited concurrency and one progress bar
    sem = asyncio.Semaphore(config.max_concurrent_tasks)

    # Initialize progress bars for each step
    tqdm = get_tqdm(use_async=True, in_notebook=in_notebook)
    with (
        tqdm(total=total_assets, desc="Step 1: Wait for Order", position=0) as wait_pbar,
        tqdm(total=total_assets, desc="Step 2: Downloading Order", position=1) as download_pbar,
        tqdm(total=total_assets, desc="Step 3: Unzip Order", position=2) as unzip_pbar,
        tqdm(total=total_assets, desc="Step 4: Cleanup Order", position=3) as cleanup_pbar,
    ):

        # Dictionary to track progress of each step
        step_progress_bars = {
            "wait_order": wait_pbar,
            "download_order": download_pbar,
            "unzip": unzip_pbar,
            "cleanup": cleanup_pbar,
        }

        # Run all tasks and collect results
        tasks = [
            asyncio.create_task(
                download_single_order(sess, order, order_idx, output_path, config, step_progress_bars, sem)
            )
            for order_idx, order, output_path in orders
        ]

        # Gather all results (None if success, tuple if failure)
        results = await asyncio.gather(*tasks)

    # Filter out successful tasks and report failures
    failures = [res for res in results if res is not None]

    if failures:
        logger.error("\n❌ Failed Tasks Summary:")
        for grid_id, step, error in failures:
            logger.error(f" - Grid {grid_id}: Failed at {step} with error: {error}")
    else:
        logger.info("All downloads processed successfully!")


# Main loop. Create order requests and download the orders when ready.
async def main_loop(
    config: DownloadConfig, save_path: Path, start_date: datetime, end_date: datetime, in_notebook: bool
) -> None:
    grid_paths = geojson_paths(config.grid_dir, in_notebook=in_notebook, check_crs=False)

    # Load the orders from disk
    all_orders = []
    for grid_path in grid_paths:
        grid_id = grid_path.stem
        grid_save_dir = save_path / grid_id

        orders = orders_to_download(grid_save_dir)
        for order_idx, order_path in orders:
            with open(order_path) as f:
                order = json.load(f)
            all_orders.append((order_idx, order, grid_save_dir))

    # Download all orders
    async with Session() as sess:
        await download_orders(sess, all_orders, config, in_notebook)


def order_download(
    config_file: Path,
    start_date: datetime,
    end_date: datetime,
):
    config, save_path = create_config(config_file, start_date=start_date, end_date=end_date)

    setup_logger(save_path, log_filename="order_download.log")

    logger.info(
        f"Ordering images for start_date={start_date} end_date={end_date} grids={config.grid_dir} to={save_path}"
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

    order_download(config_file=config_file, start_date=start_date, end_date=end_date)


if __name__ == "__main__":
    main()
