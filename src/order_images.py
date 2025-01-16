import asyncio
import json
import logging
import zipfile
from datetime import datetime
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf
from planet import OrdersClient, Session, order_request
from tqdm.asyncio import tqdm

from src.config import DownloadConfig
from src.util import geojson_paths, product_bundle_by_date, retry_task, setup_logger

logger = logging.getLogger(__name__)


def unzip_downloads(results_grid_dir: Path) -> None:
    # Location of unziped files
    order_files_dir = results_grid_dir / "files"

    # If we already unzipped the files, exit.
    if order_files_dir.exists():
        return

    # If no order.json exists then there won't be a zip file.
    order_path = results_grid_dir / "order.json"
    if not order_path.exists():
        logger.warning(f"Missing order request for {results_grid_dir.stem}")
        return

    # Get the order_id to know the name of the zip file
    with open(order_path) as f:
        order_request = json.load(f)
    order_id = str(order_request["id"])

    order_download_dir = results_grid_dir / order_id
    if not order_download_dir.exists():
        logger.warning(f"Missing order download for {results_grid_dir.stem}")
        return

    # There should just be one zip file in the download folder.
    order_download_path = list(order_download_dir.glob("*.zip"))[0]
    # Open the zip file and extract its contents
    with zipfile.ZipFile(order_download_path, "r") as zip_ref:
        zip_ref.extractall(results_grid_dir)

    assert order_files_dir.exists()


# Buid the order request including how to clip the image and how to deliver it.
def build_order_request(
    filename: str, item_ids: list[str], product_bundle: str, aoi: dict, min_acquired: datetime, config: DownloadConfig
) -> dict:

    name = f"{min_acquired.year}_{min_acquired.month}_{filename}"

    products = [order_request.product(item_ids=item_ids, product_bundle=product_bundle, item_type=config.item_type)]

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
# Skip grids that have existing order.json files.
def create_order_requests(
    grid_paths: list[Path], save_dir: Path, imagery_date: datetime, config: DownloadConfig
) -> list[tuple[dict, Path]]:
    order_requests = []

    for grid_path in grid_paths:
        grid_id = grid_path.stem

        assert grid_path.suffix == ".geojson", f"Invalid path, {grid_path.name}"

        grid_dir = save_dir / grid_id

        if (grid_dir / "order.json").exists():
            continue

        item_ids_path = grid_dir / "images_to_download.csv"
        if not item_ids_path.exists():
            logger.warning(f"Missing item download list for {grid_id}")
            continue

        udm_df = pd.read_csv(item_ids_path)
        item_ids = udm_df[udm_df.include_image]["asset_id"].tolist()
        with open(grid_path) as file:
            grid_geojson = json.load(file)

        product_bundle = product_bundle_by_date(imagery_date)
        order_request = build_order_request(
            grid_path.stem,
            item_ids,
            product_bundle,
            grid_geojson,
            imagery_date,
            config,
        )

        order_requests.append((order_request, grid_path.stem))

    return order_requests


# Create order requests and issue them for all grid paths. Save results to a file.
async def create_orders(
    sess: Session, grid_paths: list[Path], save_dir: Path, imagery_date: datetime, config: DownloadConfig
):
    # Create the order requests objects
    order_requests_to_create = create_order_requests(grid_paths, save_dir, imagery_date, config)

    logger.info(f"Starting {len(order_requests_to_create)} order requests")

    # loop through and download all the UDM2 files for the given date and grid
    try:
        order_tasks = [asyncio.create_task(create_order(sess, request)) for request, _ in order_requests_to_create]
        orders = await asyncio.gather(*order_tasks)
    except Exception as e:
        logger.error(f"Error creating order for {imagery_date}")
        logger.exception(e)
        raise e

    # Save the order results to a json file per grid
    for order, (_, grid_name) in zip(orders, order_requests_to_create):
        with open(save_dir / grid_name / "order.json", "w") as f:
            json.dump(order, f)


# Download an order zip file. If order folder already exists, skip over it.
async def download_order(
    sess: Session, order: dict, save_dir: Path, config: DownloadConfig, step_progress_bars: dict
) -> tuple[str, str, str] | None:
    grid_id = save_dir.stem
    order_id = str(order["id"])

    # Exit early if order already exists.
    if (save_dir / order_id).exists():
        for v in step_progress_bars.values():
            v.update(1)
        return

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
async def download_orders(sess: Session, orders: list[tuple[dict, Path]], config: DownloadConfig) -> None:
    logger.info(f"Downloading {len(orders)} orders")

    total_assets = len(orders)

    # Initialize progress bars for each step
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
            for order, output_path in orders
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
        logger.info("✅ All downloads processed successfully!")


# Main loop. Create order requests and download the orders when ready.
async def main_loop(config: DownloadConfig, save_path: Path, imagery_date: datetime) -> None:
    grid_paths = geojson_paths(config.grid_dir)

    async with Session() as sess:
        await create_orders(sess, grid_paths, save_path, imagery_date, config)

        logger.info("Downloading image data")

        # Load the orders from disk
        all_orders = []
        for grid_path in grid_paths:
            grid_id = grid_path.stem
            grid_dir = save_path / grid_id
            order_path = grid_dir / "order.json"
            if not order_path.exists():
                logger.warning(f"Missing order for {grid_id}")
                continue
            with open(order_path) as f:
                order = json.load(f)
                all_orders.append((order, grid_dir))

        # Download all orders
        await download_orders(sess, all_orders, config)

    # Unzip the downloads and the remove the zip file for each grid.
    for grid_path in grid_paths:
        results_grid_dir = save_path / grid_path.stem
        if not results_grid_dir.exists():
            logger.warning(f"No results directory for {grid_path.stem}")
            continue
        unzip_downloads(results_grid_dir)


@click.command()
@click.option("-c", "--config-file", type=click.Path(exists=True), required=True)
@click.option("-y", "--year", type=click.IntRange(min=1990, max=2050))
@click.option("-m", "--month", type=click.IntRange(min=1, max=12))
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

    setup_logger(save_path, log_filename="order_images.log")

    logger.info(f"Ordering images for year={year} month={month} grids={config.grid_dir} to={save_path}")

    imagery_date = datetime(year, month, 1)

    asyncio.run(main_loop(config, save_path, imagery_date))


if __name__ == "__main__":
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
