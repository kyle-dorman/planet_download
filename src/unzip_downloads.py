import json
import logging
import os
import zipfile
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf

from src.config import DownloadConfig
from src.util import setup_logger

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

    # Remove the order download directory
    os.remove(order_download_path)

    assert order_files_dir.exists()


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

    setup_logger(logger, save_path, log_filename="unzip_downloads.log")

    logger.info(f"Unzipping downloads for year={year} month={month} grids={config.grid_dir} to={save_path}")

    # Unzip the downloads and the remove the zip file for each grid.
    for grid_path in config.grid_dir.iterdir():
        results_grid_dir = save_path / grid_path.stem
        if not results_grid_dir.exists():
            logger.warning(f"No results directory for {grid_path.stem}")
            continue

        unzip_downloads(results_grid_dir)


if __name__ == "__main__":
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
