import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf
from planet import OrdersClient, Session, order_request

from src.config import DownloadConfig
from src.util import product_bundle_by_date, setup_logger

logger = logging.getLogger(__name__)


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


# Download image data
async def download_order(sess: Session, order: dict, save_dir: Path) -> None:
    logger.debug(f"Downloading order {order['id']}")
    cl = OrdersClient(sess)
    # Wait for the order to be ready
    await cl.wait(order["id"])
    # If we get here that means the order completed. Yay! Download the files.
    await cl.download_order(order["id"], directory=save_dir, overwrite=False, progress_bar=False)
    logger.debug(f"Finished downloading order {order['id']}")


# Create list of orders to create across all grid paths.
# Skip grids that have existing order.json files.
def create_order_requests(
    grid_paths: list[Path], save_dir: Path, imagery_date: datetime, config: DownloadConfig
) -> list[tuple[dict, Path]]:
    order_requests = []

    for grid_path in grid_paths:
        assert grid_path.suffix == ".geojson", f"Invalid path, {grid_path.name}"

        grid_dir = save_dir / grid_path.stem

        if (grid_dir / "order.json").exists():
            continue

        item_ids_path = grid_dir / "images_to_download.csv"
        if not item_ids_path.exists():
            logger.warning(f"Missing item download list for {grid_path.stem}")
            continue

        item_ids = pd.read_csv(item_ids_path).iloc[:, 0].tolist()
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


# Download an order and retry failed downloads a fixed number of times.
async def download_order_with_retry(sess: Session, orders: list[tuple[dict, Path]], max_retries: int) -> None:
    remaining_orders = orders

    for attempt in range(max_retries):
        logger.info(f"Attempt {attempt + 1}/{max_retries}")

        # Create tasks for all remaining orders
        download_tasks = [
            asyncio.create_task(download_order(sess, order, grid_dir)) for order, grid_dir in remaining_orders
        ]

        # Await all tasks and capture exceptions
        results = await asyncio.gather(*download_tasks, return_exceptions=True)

        # Filter out failed downloads
        remaining_orders = [
            order
            for order, result in zip(remaining_orders, results)
            if isinstance(result, Exception)  # Retain failed tasks
        ]

        # Log remaining failures
        if remaining_orders:
            logger.warning(f"Failed downloads remaining: {len(remaining_orders)}")
        else:
            logger.info("All downloads completed successfully.")
            break
    else:
        logger.error(f"Max retries reached. Remaining failed downloads: {len(remaining_orders)}")


# Main loop. Create order requests and download the orders when ready.
async def main_loop(config: DownloadConfig, save_path: Path, imagery_date: datetime) -> None:
    grid_paths = list(config.grid_dir.iterdir())

    async with Session() as sess:
        await create_orders(sess, grid_paths, save_path, imagery_date, config)

        logger.info("Downloading image data")

        # Load the orders from disk
        all_orders = []
        for grid_path in grid_paths:
            grid_dir = save_path / grid_path.stem
            order_path = grid_dir / "order.json"
            if not order_path.exists():
                logger.warning(f"Missing order for {grid_path.stem}")
                continue
            with open(order_path) as f:
                order = json.load(f)
                all_orders.append((order, grid_dir))

        # Download
        await download_order_with_retry(sess, all_orders, config.download_retries_max)


@click.command()
@click.option("-c", "--config-file", type=click.Path(exists=True), required=True)
@click.option("-m", "--month", type=int)
@click.option("-y", "--year", type=int)
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

    setup_logger(logger, save_path, log_filename="order_images.log")

    logger.info(f"Ordering images for year={year} month={month} grids={config.grid_dir} to={save_path}")

    imagery_date = datetime(year, month, 1)

    asyncio.run(main_loop(config, save_path, imagery_date))


if __name__ == "__main__":
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
