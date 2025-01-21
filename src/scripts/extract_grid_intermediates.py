import logging
from pathlib import Path

import click
from omegaconf import OmegaConf

from src.config import DownloadConfig
from src.grid import (
    create_polygon_aligned_profile_update,
    find_most_common_crs,
    open_and_convert_grid,
    reproject_and_crop_to_grid,
)
from src.util import create_config, get_tqdm, is_notebook, setup_logger, tif_paths

logger = logging.getLogger(__name__)


# Reproject and crop the UDMs that were evaluated for selection in select_udms.py
# Save the intermediates from the process.
def reproject_and_crop_udms(
    results_grid_dir: Path,
    grid_path: Path,
    config: DownloadConfig,
) -> None:
    udm_paths = tif_paths(results_grid_dir / "udm")

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
    logger.info("Reprojecting all UDMs")
    tqdm = get_tqdm(use_async=False, in_notebook=is_notebook())
    for udm_path in tqdm(udm_paths):
        cropped_path = cropped_dir / udm_path.name
        reprojected_path = reprojected_udm_dir / udm_path.name
        if cropped_path.exists() and reprojected_path.exists():
            continue

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
        logger.info(f"Reprojecting selected {name}s")
        tqdm = get_tqdm(use_async=False, in_notebook=is_notebook())
        for tif_path in tqdm(file_paths):
            cropped_path = cropped_dir / tif_path.name
            reprojected_path = reprojected_udm_dir / tif_path.name

            if cropped_path.exists() and reprojected_path.exists():
                continue

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


def extract_grid_intermediates(
    config_file: Path,
    grid_id: str,
    year: int,
    month: int,
) -> None:
    config, save_path = create_config(config_file, year=year, month=month)

    setup_logger()

    grid_path = config.grid_dir / f"{grid_id}.geojson"
    results_grid_dir = save_path / grid_id

    # Save the configuration to a YAML file
    OmegaConf.save(config, results_grid_dir / "config.yaml")

    reproject_and_crop_download_outputs(results_grid_dir, grid_path, config)
    reproject_and_crop_udms(results_grid_dir, grid_path, config)


# Create the intermediate outputs neccessary to inspect a grid's UDMs and downloaded data
@click.command()
@click.option("-c", "--config-file", type=click.Path(exists=True), required=True)
@click.option("-g", "--grid-id", type=str)
@click.option("-y", "--year", type=click.IntRange(min=1990, max=2050))
@click.option("-m", "--month", type=click.IntRange(min=1, max=12))
def main(
    config_file: Path,
    grid_id: str,
    year: int,
    month: int,
):
    config_file = Path(config_file)

    extract_grid_intermediates(config_file=config_file, grid_id=grid_id, month=month, year=year)

    logger.info("Done!")


if __name__ == "__main__":
    main()
