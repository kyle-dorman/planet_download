import logging
import shutil
from datetime import datetime
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

from src.util import (
    check_and_create_env,
    create_config,
    geojson_paths,
    get_tqdm,
    is_notebook,
    setup_logger,
)

logger = logging.getLogger(__name__)


def udm_cleanup(
    config_file: Path,
    start_date: datetime,
    end_date: datetime,
):
    config, save_path = create_config(config_file, start_date=start_date, end_date=end_date)

    setup_logger(save_path, log_filename="udm_cleanup.log")

    logger.info(f"Cleaning up UDMs for start_date={start_date} end_date={end_date} grids={config.grid_dir}")

    if not config.cleanup_udm:
        logger.info("Cleanup OFF")
        return

    in_notebook = is_notebook()

    grid_paths = geojson_paths(config.grid_dir, in_notebook=in_notebook, check_crs=False)

    tqdm = get_tqdm(use_async=False, in_notebook=in_notebook)
    for grid_path in tqdm(grid_paths):
        grid_id = grid_path.stem
        udm_dir = save_path / grid_id / "udm"

        if udm_dir.exists():
            shutil.rmtree(udm_dir)


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

    udm_cleanup(config_file=config_file, start_date=start_date, end_date=end_date)


if __name__ == "__main__":
    main()
