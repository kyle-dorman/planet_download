import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf

from src.config import DownloadConfig
from src.grid import (
    create_polygon_aligned_profile_update,
    find_most_common_crs,
    open_and_convert_grid,
    reproject_and_crop_to_grid,
)
from src.util import setup_logger

logger = logging.getLogger(__name__)


# Reproject and crop the UDMs that were evaluated for selection in select_udms.py
# Save the intermediates from the process.
def reproject_and_crop_udms(
    results_grid_dir: Path,
    grid_path: Path,
    config: DownloadConfig,
) -> None:
    udm_paths = list((results_grid_dir / "udm").iterdir())

    # Choose CRS to work in
    crs = find_most_common_crs(udm_paths)

    # Load Target Grid and convert grid to base udm crs
    grid_transformed = open_and_convert_grid(grid_path, crs)

    # Create the new consistent grid
    profile_update = create_polygon_aligned_profile_update(grid_transformed, crs, config.ground_sample_distance)

    # Save reprojected and cropped intermediates
    cropped_dir = results_grid_dir / "udm_cropped"
    cropped_dir.mkdir(exist_ok=True)
    reprojected_udm_dir = results_grid_dir / "udm_reprojected"
    reprojected_udm_dir.mkdir(exist_ok=True)

    # Crop the UDMs
    logger.info("Saving cropped UDMs")
    for udm_path in udm_paths:
        cropped_path = cropped_dir / udm_path.name
        reprojected_path = reprojected_udm_dir / udm_path.name
        reproject_and_crop_to_grid(
            tif_path=udm_path,
            grid_geom=grid_transformed,
            profile_update=profile_update,
            repro_path=reprojected_path,
            out_path=cropped_path,
            channels=1,
        )


# Reproject and crop the download outputs so they are in a consistent grid and can be analyzed in the notebook
def reproject_and_crop_download_outputs(results_grid_dir: Path, grid_path: Path, config: DownloadConfig) -> None:
    order_files_dir = results_grid_dir / "files"

    assert order_files_dir.exists(), f"Missing order files directory {order_files_dir}"

    asset_file_paths = list(order_files_dir.glob("*AnalyticMS*.tif"))
    udm_file_paths = list(order_files_dir.glob("*udm2*.tif"))

    udm_crs = crs = find_most_common_crs(udm_file_paths)
    asset_crs = find_most_common_crs(asset_file_paths)
    assert udm_crs == asset_crs, "Exected UDM CRS to match Asset CRS"

    # Load Target Grid and convert grid to base udm crs
    grid_transformed = open_and_convert_grid(grid_path, crs)

    # Create the new consistent grid
    profile_update = create_polygon_aligned_profile_update(grid_transformed, crs, config.ground_sample_distance)

    for file_paths, name in [(udm_file_paths, "udm"), (asset_file_paths, "asset")]:
        # Save reprojected and cropped intermediates
        cropped_dir = results_grid_dir / f"files_{name}_cropped"
        cropped_dir.mkdir(exist_ok=True)
        reprojected_udm_dir = results_grid_dir / f"files_{name}_reprojected"
        reprojected_udm_dir.mkdir(exist_ok=True)

        # Crop the UDMs
        logger.info(f"Cropping {name}s")
        for tif_path in file_paths:
            cropped_path = cropped_dir / tif_path.name
            reprojected_path = reprojected_udm_dir / tif_path.name

            # Only crop the first channel for the UDM, otherwise all the channels
            channels = 1 if name == "udm" else None
            reproject_and_crop_to_grid(
                tif_path=tif_path,
                grid_geom=grid_transformed,
                profile_update=profile_update,
                repro_path=reprojected_path,
                out_path=cropped_path,
                channels=channels,
            )


# Create the intermediate outputs neccessary to inspect a grid's UDMs and downloaded data
@click.command()
@click.option("-c", "--config-file", type=click.Path(exists=True), required=True)
@click.option("-g", "--grid-id", type=str)
@click.option("-m", "--month", type=int)
@click.option("-y", "--year", type=int)
def main(
    config_file: Path,
    grid_id: str,
    month: int,
    year: int,
):
    config_file = Path(config_file)
    base_config = OmegaConf.structured(DownloadConfig)
    override_config = OmegaConf.load(config_file)
    config: DownloadConfig = OmegaConf.merge(base_config, override_config)  # type: ignore

    save_path = config.save_dir / str(year) / str(month).zfill(2)
    save_path.mkdir(exist_ok=True, parents=True)

    setup_logger(logger)

    grid_path = config.grid_dir / f"{grid_id}.geojson"
    results_grid_dir = save_path / grid_id

    # Save the configuration to a YAML file
    OmegaConf.save(config, results_grid_dir / "config.yaml")

    reproject_and_crop_download_outputs(results_grid_dir, grid_path, config)
    reproject_and_crop_udms(results_grid_dir, grid_path, config)


if __name__ == "__main__":
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
