import logging
import shutil
from datetime import datetime
from pathlib import Path

import click
import tqdm

from src.util import create_config, setup_logger

logger = logging.getLogger(__name__)


def copy_to_process_dir(
    config_file: Path,
    start_date: datetime,
) -> None:
    config, save_path = create_config(config_file, start_date=start_date)

    setup_logger()

    if config.processing_dir is None:
        logger.warning("Processing directory is None. Skipping...")
        return

    logger.info(f"Copying files to the processing directory {config.processing_dir}!")

    for grid_dir in tqdm.tqdm(save_path.iterdir()):
        if not grid_dir.is_dir():
            continue

        grid_id = grid_dir.name
        processing_grid_dir = config.processing_dir / save_path.parent.parent / save_path.parent / grid_id
        surface_reflectance_dir = processing_grid_dir / "inputData" / "surfaceReflectance"
        udm_dir = processing_grid_dir / "inputData" / "udm"
        surface_reflectance_dir.mkdir(exist_ok=True, parents=True)
        udm_dir.mkdir(exist_ok=True, parents=True)

        order_files_dir = grid_dir / "files"

        assert order_files_dir.exists(), f"Missing order files directory {order_files_dir}"

        asset_file_paths = list(order_files_dir.glob("*AnalyticMS*.tif"))
        udm_file_paths = list(order_files_dir.glob("*udm2*.tif"))

        for asset_file_path in asset_file_paths:
            shutil.copy(asset_file_path, surface_reflectance_dir / asset_file_path.name)

        for udm_file_path in udm_file_paths:
            shutil.copy(udm_file_path, surface_reflectance_dir / udm_file_path.name)

    logger.info("Done!")


# Copy UDM and reflectance files to directory for downstream processing
@click.command()
@click.option("-c", "--config-file", type=click.Path(exists=True), required=True)
@click.option(
    "--start-date", type=click.DateTime(formats=["%Y-%m-%d"]), help="Start date in YYYY-MM-DD format.", required=True
)
@click.option(
    "--end-date", type=click.DateTime(formats=["%Y-%m-%d"]), help="End date in YYYY-MM-DD format.", required=True
)
def main(
    config_file: Path,
    start_date: datetime,
    end_date: datetime,
):
    config_file = Path(config_file)

    copy_to_process_dir(config_file=config_file, start_date=start_date)


if __name__ == "__main__":
    main()
