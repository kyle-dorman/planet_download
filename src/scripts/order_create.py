import asyncio
import json
import logging
from datetime import datetime
from itertools import islice
from pathlib import Path
from uuid import uuid4

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
    log_structured_failure,
    retry_task,
    run_async_function,
    setup_logger,
)

logger = logging.getLogger(__name__)
CATEGORY = Path(__file__).stem


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
    paths = sorted(list(grid_dir.glob("order_*.json")) + list(grid_dir.glob("order.json")))
    return [p for p in paths if "request" not in p.name]


# Buid the order request including how to clip the image and how to deliver it.
def build_region_order_request(
    filename: str,
    item_ids: list[str],
    product_bundle: str,
    aoi: dict,
    start_date: datetime,
    end_date: datetime,
    config: DownloadConfig,
) -> dict:

    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    name = f"{start_date_str}_{end_date_str}_{filename}"

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
async def create_region_order(
    sess: Session,
    order_request: dict,
    order_idx: int,
    results_grid_dir: Path,
    config: DownloadConfig,
    run_id: str,
    start_date: datetime,
    end_date: datetime,
    save_dir,
    sem: asyncio.Semaphore,
) -> None:
    async with sem:
        logger.debug(f"Creating order {order_request['name']}")
        cl = OrdersClient(sess)
        grid_id = results_grid_dir.stem

        # Schedule the orders
        # Wait for the order to be ready
        async def create_order_retry():
            return await cl.create_order(order_request)

        try:
            order = await retry_task(create_order_retry, config.download_retries_max, config.download_backoff)

            # Save order
            with open(results_grid_dir / f"order_{order_idx}.json", "w") as f:
                json.dump(order, f)

            logger.debug(f"Finished creating order {order_request['name']}")
        except Exception as error:
            log_structured_failure(
                save_path=save_dir,
                run_id=run_id,
                category=CATEGORY,
                payload={
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "grid_id": grid_id,
                    "error": repr(error),
                    "error_type": type(error).__name__,
                    "error_args": error.args,
                    "order_idx": order_idx,
                },
            )


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

        grid_save_dir = save_dir / grid_id

        # If the order_*.json file exists, then we have already scheduled this order.
        if any(get_order_jsons(grid_save_dir)):
            continue

        item_ids_path = grid_save_dir / config.udm_select_file_name
        if not item_ids_path.exists():
            logger.debug(f"Missing item download list for {grid_id}")
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
            order_request = build_region_order_request(
                f"{grid_path.stem}_{idx}",
                item_batch,
                product_bundle,
                grid_geojson,
                start_date,
                end_date,
                config,
            )
            with open(grid_save_dir / f"order_request_{idx}.json", "w") as f:
                json.dump(order_request, f)

            order_requests.append((order_request, idx, grid_save_dir))

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
    run_id: str,
):
    # Create the order requests objects
    logger.info("Creating order requests")
    order_requests = create_order_requests(grid_paths, save_dir, start_date, end_date, config, in_notebook)

    logger.info(f"Starting {len(order_requests)} order requests")

    # download items with limited concurrency and one progress bar
    sem = asyncio.Semaphore(config.max_concurrent_tasks)

    try:
        order_tasks = [
            asyncio.create_task(
                create_region_order(
                    sess=sess,
                    order_request=order_request,
                    order_idx=order_idx,
                    results_grid_dir=results_grid_dir,
                    config=config,
                    run_id=run_id,
                    start_date=start_date,
                    end_date=end_date,
                    save_dir=save_dir,
                    sem=sem,
                )
            )
            for order_request, order_idx, results_grid_dir in order_requests
        ]
        await asyncio.gather(*order_tasks)
    except Exception as e:
        logger.error(f"Error creating order for {start_date} {end_date}")
        logger.exception(e)
        raise e


# Main loop. Create order requests and download the orders when ready.
async def main_loop(
    config: DownloadConfig,
    save_path: Path,
    start_date: datetime,
    end_date: datetime,
    in_notebook: bool,
    run_id: str,
) -> None:
    grid_paths = geojson_paths(config.grid_dir, in_notebook=in_notebook, check_crs=False)

    async with Session() as sess:
        await create_orders(sess, grid_paths, save_path, start_date, end_date, config, in_notebook, run_id)


def order_create(
    config_file: Path,
    start_date: datetime,
    end_date: datetime,
):
    config, save_path = create_config(config_file, start_date=start_date, end_date=end_date)

    setup_logger(save_path, log_filename="order_create.log")

    run_id = uuid4().hex
    logger.info(
        f"Run id={run_id} Create image orders for start_date={start_date} end_date={end_date} "
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

    order_create(config_file=config_file, start_date=start_date, end_date=end_date)


if __name__ == "__main__":
    main()
