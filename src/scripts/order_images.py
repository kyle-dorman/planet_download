import asyncio
import json
import logging
import zipfile
from datetime import datetime
from itertools import islice
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from planet import OrdersClient, Session, order_request

from src.config import DownloadConfig, product_bundle_string
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


def batched(iterable, size):
    # Patch for python 3.12 batched function
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk


def get_order_jsons(grid_dir: Path) -> list[Path]:
    # Get order json files. Looks for legacy order.json as well as batched order_*.json
    return list(grid_dir.glob("order_*.json")) + list(grid_dir.glob("order.json"))


def unzip_downloads(results_grid_dir: Path) -> None:
    # Location of unziped files
    order_files_dir = results_grid_dir / "files"

    # If we already unzipped the files, exit.
    if order_files_dir.exists():
        return

    # If no order_*.json exists then there won't be a zip file.
    order_paths = get_order_jsons(results_grid_dir)
    if not len(order_paths):
        logger.warning(f"Missing order request for {results_grid_dir.stem}")
        return

    # Get the order_id to know the name of the zip file
    for order_path in order_paths:
        with open(order_path) as f:
            order_request = json.load(f)

        order_id = str(order_request["id"])

        order_download_dir = results_grid_dir / order_id
        if not order_download_dir.exists():
            logger.warning(f"Missing order download for {results_grid_dir.stem} {order_path.name}")
            continue

        # There should just be one zip file in the download folder.
        order_download_path = list(order_download_dir.glob("*.zip"))[0]
        # Open the zip file and extract its contents
        with zipfile.ZipFile(order_download_path, "r") as zip_ref:
            zip_ref.extractall(results_grid_dir)

        assert order_files_dir.exists()


# Buid the order request including how to clip the image and how to deliver it.
def build_order_request(
    filename: str,
    item_ids: list[str],
    product_bundle: str,
    aoi: dict,
    start_date: datetime,
    end_date: datetime,
    config: DownloadConfig,
) -> dict:

    name = f"{start_date}_{end_date}_{filename}"

    products = [
        order_request.product(item_ids=item_ids, product_bundle=product_bundle, item_type=config.item_type.value)
    ]

    tools = [order_request.clip_tool(aoi)]

    delivery = order_request.delivery(archive_type="zip", single_archive=True, archive_filename=f"{name}.zip")

    return order_request.build_request(
        name=name,
        products=products,
        tools=tools,
        delivery=delivery,
    )


# Create an order for image data
async def create_order(sess: Session, request: dict) -> dict:
    logger.debug(f"Creating order {request['name']}")
    cl = OrdersClient(sess)
    order = await cl.create_order(request)
    logger.debug(f"Finished creating order {request['name']}")

    return order


# Create list of orders to create across all grid paths.
# Skip grids that have existing order_*.json files.
def create_order_requests(
    grid_paths: list[Path],
    save_dir: Path,
    start_date: datetime,
    end_date: datetime,
    config: DownloadConfig,
    in_notebook: bool,
) -> list[tuple[dict, int, Path]]:
    order_requests = []

    tqdm = get_tqdm(use_async=False, in_notebook=in_notebook)
    for grid_path in tqdm(grid_paths):
        grid_id = grid_path.stem

        assert grid_path.suffix == ".geojson", f"Invalid path, {grid_path.name}"

        grid_dir = save_dir / grid_id

        # If the order_*.json file exists, then we have already scheduled this order.
        if any(get_order_jsons(grid_dir)):
            continue

        item_ids_path = grid_dir / "images_to_download.csv"
        if not item_ids_path.exists():
            logger.warning(f"Missing item download list for {grid_id}")
            continue

        # Get the list of item_ids to download
        udm_df = pd.read_csv(item_ids_path)
        item_ids = udm_df[udm_df.include_image]["asset_id"].tolist()

        if not len(item_ids):
            logger.warning(f"No valid images for {grid_id}. Skipping...")
            continue

        # Load the grid AOI
        with open(grid_path) as file:
            grid_geojson = json.load(file)

        product_bundle = product_bundle_string(config)

        # Create order requests
        for idx, item_batch in enumerate(batched(item_ids, config.order_item_limit)):
            order_request = build_order_request(
                f"{grid_path.stem}_{idx}",
                item_batch,
                product_bundle,
                grid_geojson,
                start_date,
                end_date,
                config,
            )

            order_requests.append((order_request, idx, grid_path.stem))

    return order_requests


# Create order requests and issue them for all grid paths. Save results to a file.
async def create_orders(
    sess: Session,
    grid_paths: list[Path],
    save_dir: Path,
    start_date: datetime,
    end_date: datetime,
    config: DownloadConfig,
    in_notebook: bool,
):
    # Create the order requests objects
    logger.info("Creating order requests")
    order_requests_to_create = create_order_requests(grid_paths, save_dir, start_date, end_date, config, in_notebook)

    logger.info(f"Starting {len(order_requests_to_create)} order requests")

    # Schedule the orders
    try:
        order_tasks = [asyncio.create_task(create_order(sess, request)) for request, _, _ in order_requests_to_create]
        orders = await asyncio.gather(*order_tasks)
    except Exception as e:
        logger.error(f"Error creating order for {start_date} {end_date}")
        logger.exception(e)
        raise e

    logger.info("Saving orders")

    # Save the order results to a json file per grid
    for order, (_, idx, grid_name) in zip(orders, order_requests_to_create):
        with open(save_dir / grid_name / f"order_{idx}.json", "w") as f:
            json.dump(order, f)


# Download an order zip file. If order folder already exists, skip over it.
async def download_order(
    sess: Session, order: dict, save_dir: Path, config: DownloadConfig, step_progress_bars: dict
) -> tuple[str, str, str] | None:
    grid_id = save_dir.stem
    order_id = str(order["id"])

    cl = OrdersClient(sess)

    # Wait for the order to be ready
    async def wait_order():
        await cl.wait(order_id)

    try:
        await retry_task(wait_order, config.download_retries_max, config.download_backoff)
    except Exception as e:
        return (grid_id, "wait_order", str(e))
    step_progress_bars["wait_order"].update(1)

    # Download the files
    async def download_order():
        await cl.download_order(order_id, directory=save_dir, overwrite=False, progress_bar=False)

    try:
        await retry_task(download_order, config.download_retries_max, config.download_backoff)
    except Exception as e:
        return (grid_id, "download_order", str(e))
    step_progress_bars["download_order"].update(1)


# Download an order and retry failed downloads a fixed number of times.
async def download_orders(
    sess: Session, orders: list[tuple[dict, Path]], config: DownloadConfig, in_notebook: bool
) -> None:
    # Skip orders that were previously downloaded
    orders_to_download = []
    for order, output_path in orders:
        order_id = str(order["id"])
        order_dir = output_path / order_id
        # Exit early if order already exists.
        if order_dir.exists():
            continue
        orders_to_download.append((order, output_path))

    total_assets = len(orders_to_download)
    logger.info(f"Downloading {total_assets} orders")

    # Initialize progress bars for each step
    tqdm = get_tqdm(use_async=True, in_notebook=in_notebook)
    with (
        tqdm(total=total_assets, desc="Step 1: Wait for Order", position=0) as wait_pbar,
        tqdm(total=total_assets, desc="Step 2: Downloading Order", position=1) as download_pbar,
    ):

        # Dictionary to track progress of each step
        step_progress_bars = {
            "wait_order": wait_pbar,
            "download_order": download_pbar,
        }

        # Run all tasks and collect results
        tasks = [
            asyncio.create_task(download_order(sess, order, output_path, config, step_progress_bars))
            for order, output_path in orders_to_download
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
    grid_paths = geojson_paths(config.grid_dir)

    async with Session() as sess:
        await create_orders(sess, grid_paths, save_path, start_date, end_date, config, in_notebook)

        logger.debug("Downloading image data")

        # Load the orders from disk
        all_orders = []
        for grid_path in grid_paths:
            grid_id = grid_path.stem
            grid_dir = save_path / grid_id
            order_paths = get_order_jsons(grid_dir)
            if not any(order_paths):
                logger.warning(f"Missing order for {grid_id}")
                continue
            for order_path in order_paths:
                with open(order_path) as f:
                    order = json.load(f)
                    all_orders.append((order, grid_dir))

        # Download all orders
        await download_orders(sess, all_orders, config, in_notebook)

    logger.info("Unzipping Downloads")

    # Unzip the downloads and the remove the zip file for each grid.
    tqdm = get_tqdm(use_async=False, in_notebook=in_notebook)
    for grid_path in tqdm(grid_paths, dynamic_ncols=True):
        results_grid_dir = save_path / grid_path.stem
        if not results_grid_dir.exists():
            logger.warning(f"No results directory for {grid_path.stem}")
            continue
        unzip_downloads(results_grid_dir)


def order_images(
    config_file: Path,
    start_date: datetime,
    end_date: datetime,
):
    config, save_path = create_config(config_file, start_date=start_date, end_date=end_date)

    setup_logger(save_path, log_filename="order_images.log")

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

    order_images(config_file=config_file, start_date=start_date, end_date=end_date)


if __name__ == "__main__":
    main()
