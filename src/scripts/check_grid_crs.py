import json
import logging
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import click
import geopandas as gpd
from pyproj import CRS
from pyproj.exceptions import CRSError

from src.util import create_config, geojson_paths, get_tqdm, is_notebook, setup_logger

logger = logging.getLogger(__name__)


def explicit_geojson_crs_name(geojson_path: Path) -> str | None:
    with geojson_path.open(encoding="utf-8") as f:
        geojson = json.load(f)

    crs = geojson.get("crs")
    if not isinstance(crs, dict):
        return None

    properties = crs.get("properties")
    if not isinstance(properties, dict):
        return None

    name = properties.get("name")
    if not name:
        return None

    return str(name)


def has_crs(geojson_path: Path) -> None:
    crs_name = explicit_geojson_crs_name(geojson_path)
    if crs_name is None:
        raise ValueError(f"{geojson_path} is missing an explicit GeoJSON CRS")

    try:
        CRS.from_user_input(crs_name)
    except CRSError as error:
        raise ValueError(f"{geojson_path} has an invalid CRS: {crs_name}") from error

    grid = gpd.read_file(geojson_path)
    if grid.crs is None:
        raise ValueError(f"{geojson_path} is missing a CRS")


def crs_check_error(geojson_path: Path) -> tuple[Path, str] | None:
    try:
        has_crs(geojson_path)
    except ValueError as error:
        return geojson_path, str(error)

    return None


def check_all_has_crs(paths: list[Path], workers: int, in_notebook: bool) -> None:
    """
    Parallelize CRS checks over a list of Path objects.
    Checks every path, then raises if any are missing CRS metadata.
    """
    if not paths:
        logger.info("No GeoJSON files found to check.")
        return

    errors: list[tuple[Path, str]] = []
    worker_count = max(1, min(workers, len(paths)))
    this_tqdm = get_tqdm(use_async=False, in_notebook=in_notebook)
    # use fork instead of spawn to avoid semaphore leaks on macOS
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=worker_count) as pool:
        for error in this_tqdm(
            pool.imap_unordered(crs_check_error, paths, chunksize=100),
            total=len(paths),
            desc="Checking CRS",
        ):
            if error is not None:
                errors.append(error)

    if errors:
        error_list = "\n".join(f"- {path}: {message}" for path, message in sorted(errors))
        raise RuntimeError(f"{len(errors)} GeoJSON file(s) failed CRS validation:\n{error_list}")

    logger.info(f"All {len(paths)} files have a CRS.")


def check_grid_crs(config_file: Path, start_date: datetime, end_date: datetime) -> None:
    config, save_path = create_config(config_file, start_date=start_date, end_date=end_date)

    setup_logger(save_path, log_filename="check_grid_crs.log")

    logger.info(f"Checking CRS for grids={config.grid_dir}")

    in_notebook = is_notebook()
    grid_paths = geojson_paths(config.grid_dir)
    check_all_has_crs(grid_paths, workers=mp.cpu_count(), in_notebook=in_notebook)


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
) -> None:
    check_grid_crs(config_file=Path(config_file), start_date=start_date, end_date=end_date)


if __name__ == "__main__":
    main()
