import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

import click
from dateutil.relativedelta import relativedelta
from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf
from planet import DataClient, Session, data_filter

from src.config import DownloadConfig
from src.util import asset_type_by_date, setup_logger

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
    asset_type = asset_type_by_date(min_acquired)
    superdove_filter = data_filter.asset_filter([asset_type])

    # Has ground control points
    ground_control_filter = data_filter.string_in_filter("ground_control", [str(config.ground_control).lower()])

    # cloud cover filter
    # TOD0: Does this even do anything?
    cloud_cover_filter = data_filter.range_filter("cloud_cover", lte=config.cloud_cover)

    # set processing level filter
    processing_filter = data_filter.string_in_filter("publishing_stage", [config.publishing_stage])

    # combine all of the filters
    return data_filter.and_filter(
        [geom_filter, date_range_filter, superdove_filter, ground_control_filter, cloud_cover_filter, processing_filter]
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
        "features": [{"type": "Feature", "properties": {}, "geometry": item["geometry"]} for item in item_list],
    }

    with open(save_path, "w") as f:
        json.dump(geoms, f)

    logger.debug(f"Saved search geometry to {save_path}")


# Asynchronously downloads a single udm asset for the given item.
# Activates and waits for the asset to be ready before downloading it to the specified path.
# Skips files that already exist.
async def download_udm(item: dict, sess: Session, output_path: Path, config: DownloadConfig) -> None:
    if any(pth.stem.startswith(item["id"]) for pth in output_path.iterdir()):
        return
    logger.debug(f"Downloading UDM - {output_path.parent.name} {item['id']}")

    cl = DataClient(sess)

    # Get Asset
    asset_desc = await cl.get_asset(
        item_type_id=item["properties"]["item_type"], item_id=item["id"], asset_type_id=config.udm_asset_type
    )

    # Activate Asset
    await cl.activate_asset(asset=asset_desc)
    # Wait for Asset
    asset = await cl.wait_asset(asset=asset_desc)

    # Download Asset
    asset_path = await cl.download_asset(asset=asset, directory=output_path, overwrite=False, progress_bar=False)
    asset_path = Path(asset_path)

    logger.debug(f"Downloaded UDM - {output_path.parent.name} {item['id']}")


# Asynchronously downloads all udm assets for the given list of items.
async def download_all_udms(item_lists: list[tuple[dict, Path]], sess: Session, config: DownloadConfig) -> None:
    logger.info(f"Downloading {len(item_lists)} udm items")

    tasks = [asyncio.create_task(download_udm(item, sess, output_path, config)) for item, output_path in item_lists]
    await asyncio.gather(*tasks)

    logger.debug("Downloaded all udm items")


# Gets a list of all UDMs which need to be downloaded across all grids.
async def get_download_list(
    sess: Session, config: DownloadConfig, save_path: Path, imagery_date: datetime
) -> list[tuple[dict, Path]]:
    to_download = []
    for grid_path in config.grid_dir.iterdir():
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

            logger.info(f"Found {len(item_list)} matching grids for {grid_path.stem}")

            # Save the results
            with open(search_results_path, "w") as f:
                json.dump(item_list, f)
            save_search_geom(item_list, grid_save_path / "search_geometries.json")

        if len(item_list) == 0:
            logger.warning(f"No matches for {grid_path.stem} {imagery_date}")
            continue

        for item in item_list:
            to_download.append((item, udm_save_dir))

    return to_download


# Main loop. Download all overlapping UDMs for a given date and directory of grids.
async def main_loop(config: DownloadConfig, save_path: Path, imagery_date: datetime) -> None:
    async with Session() as sess:
        to_download = await get_download_list(sess, config, save_path, imagery_date)

        # loop through and download all the UDM2 files for the given date and grid
        try:
            await download_all_udms(to_download, sess, config)
        except Exception as e:
            logger.error(f"Error downloading udm data for {imagery_date}")
            logger.exception(e)


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

    assert config.grid_dir.exists(), f"grid_dir {config.grid_dir} does not exist!"

    save_path = config.save_dir / str(year) / str(month).zfill(2)
    save_path.mkdir(exist_ok=True, parents=True)

    # Save the configuration to a YAML file
    OmegaConf.save(config, save_path / "config.yaml")

    setup_logger(logger, save_path, log_filename="download_udms.log")

    imagery_date = datetime(year, month, 1)

    logger.info(f"Downloading UDMs for year={year} month={month} grids={config.grid_dir} to={save_path}")

    asyncio.run(main_loop(config, save_path, imagery_date))


if __name__ == "__main__":
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
