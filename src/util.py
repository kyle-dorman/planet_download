import asyncio
import logging
import multiprocessing as mp
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Type

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.errors
from matplotlib.dates import relativedelta
from omegaconf import OmegaConf
from tqdm.asyncio import tqdm_asyncio
from tqdm.notebook import tqdm_notebook
from tqdm.std import tqdm

from src.config import DownloadConfig, validate_config

logger = logging.getLogger(__name__)


def parse_acquisition_datetime(filepath: Path) -> datetime:
    """
    #: Regular expression used to extract acquisition datetime from Planet filename.
    # <acquisition date>_<acquisition time>_<...>.<extension>
    #
    # Examples:
    # - 20221001_175121_27_2460_3B_udm2.tif
    # - 20190312_181227_103a_3B_udm2.tif
    # - 20200223_163753_1_0f49_3B_udm2.tif
    # - 20190630_214558_ssc8_u0003_pansharpened_udm2.tif
    """

    pattern = re.compile(
        r"""
        (?P<date>\d{8})_        # YYYYMMDD
        (?P<time>\d{6})         # HHMMSS
    """,
        re.VERBOSE,
    )

    match = pattern.search(filepath.stem)
    assert match is not None, f"Could not parse filename {filepath.stem}"

    return datetime.strptime(match.group("date") + match.group("time"), "%Y%m%d%H%M%S")


def tif_paths(directory: Path) -> list[Path]:
    all_paths = sorted([pth for pth in directory.iterdir() if pth.suffix == ".tif"])
    paths = []
    for pth in all_paths:
        try:
            with rasterio.open(pth):
                paths.append(pth)
        except rasterio.errors.RasterioIOError as e:
            logger.exception(e)

    return paths


def geojson_paths(directory: Path, in_notebook: bool, check_crs: bool) -> list[Path]:
    """Get geojson files in a directory

    Args:
        directory (Path): The directory to look in

    Returns:
        list[Path]: A list of paths. Paths are validated for CRS.
    """
    logger.info("Finding grids")
    dir_search = ""

    current = next((p for p in directory.iterdir() if p.is_dir()), None)
    while current is not None:
        dir_search += "*/"
        current = next((p for p in current.iterdir() if p.is_dir()), None)

    paths = sorted(list(directory.glob(f"{dir_search}*.geojson")))
    logger.info(f"Found {len(paths)} grids")

    if check_crs:
        check_all_has_crs(paths, workers=mp.cpu_count(), in_notebook=in_notebook)

    return paths


# strip the _3B_udm2 from the file name
# e.g. 20230901_182511_53_2486_3B_udm2.tif
def cleaned_asset_id(filepath: Path) -> str:
    if "3B_udm2_clip" in filepath.stem:
        # 20200223_163753_1_0f49_3B_udm2_clip.tif
        return "_".join(filepath.stem.split("_")[:-3])
    elif "3B_udm2" in filepath.stem:
        # 20200223_163753_1_0f49_3B_udm2.tif
        return "_".join(filepath.stem.split("_")[:-2])
    elif "pansharpened_udm2" in filepath.stem:
        # 20190630_214558_ssc8_u0003_pansharpened_udm2
        return "_".join(filepath.stem.split("_")[:-2])
    else:
        raise RuntimeError(f"Unexpected asset name {filepath.stem}")


# Retry wrapper around an async function that may fail
async def retry_task(task_func, retries: int, retry_delay: float) -> Any:
    attempt = 0
    while attempt < retries:
        try:
            return await task_func()
        except Exception as e:
            attempt += 1
            if attempt < retries:
                wait_time = retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                await asyncio.sleep(wait_time)
            else:
                raise e  # Return the error after max retries


def setup_logger(save_dir: Path | None = None, log_filename: str = "log.log"):
    # Configure third-party loggers
    logging.getLogger("planet").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to prevent duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Create a formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if a save directory is provided)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        file_handler = logging.FileHandler(save_dir / log_filename)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Confirm setup
    root_logger.info("Logger initialized. Logging to console%s.", f" and {save_dir / log_filename}" if save_dir else "")


def write_env_file(api_key: str, env_path: Path = Path(".env")) -> None:
    """
    Writes the provided API key to a .env file with the variable name PL_API_KEY.
    """
    with env_path.open("w") as file:
        file.write(f"PL_API_KEY={api_key}\n")
    logger.info(f"API key saved to {env_path}")


def check_and_create_env(env_path: Path = Path(".env")) -> None:
    """
    Checks if the .env file exists. If not, prompts the user for an API key and writes it to the file.
    """
    if env_path.exists():
        logger.info(f"🔎 {env_path} already exists. No action needed.")
    else:
        api_key = input("Enter your Planet API key: ").strip()
        assert api_key, "Must pass an API Key!"

        write_env_file(api_key, env_path)


def create_config(config_file: Path, start_date: datetime, end_date: datetime) -> tuple[DownloadConfig, Path]:
    base_config = OmegaConf.structured(DownloadConfig)
    override_config = OmegaConf.load(config_file)
    config: DownloadConfig = OmegaConf.merge(base_config, override_config)  # type: ignore

    validate_config(config)

    assert config.grid_dir.exists(), f"grid_dir {config.grid_dir} does not exist!"

    save_path = config.save_dir / str(start_date.year)

    # If date range is smaller than this delta, add to the folder path
    deltas = [
        (relativedelta(years=1), str(start_date.month)),
        (relativedelta(months=1), str(start_date.day)),
    ]
    for delta, folder_name in deltas:
        if end_date < start_date + delta:
            save_path = save_path / folder_name

    save_path.mkdir(exist_ok=True, parents=True)

    # Save the configuration to a YAML file
    OmegaConf.save(config, save_path / "config.yaml")

    return config, save_path


def run_async_function(coro):
    """
    Runs an async function safely in both standard scripts and Jupyter Notebooks.

    Args:
        coro: The coroutine to run.
    """
    try:
        # If no event loop is running, run normally
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No event loop running, safe to use asyncio.run()
        asyncio.run(coro)
    else:
        # Event loop is running (e.g., in Jupyter), so use create_task
        task = loop.create_task(coro)
        return task  # Optional: return task if caller wants to await it


# Determine if we are in a notebook or not. Only works outside an eventloop.
def is_notebook() -> bool:
    try:
        # If no event loop is running, run normally
        asyncio.get_running_loop()
    except RuntimeError:
        # No event loop running, we are NOT in a notebook
        return False
    else:
        return True


# Load the correct progress bar class based on the run context (notebook vs CLI)
def get_tqdm(use_async: bool, in_notebook: bool) -> Type[tqdm]:
    if in_notebook:
        return tqdm_notebook
    elif use_async:
        return tqdm_asyncio
    else:
        return tqdm


def has_crs(geojson_path: Path) -> None:
    """Verify geojson file has a CRS

    Args:
        geojson_path (Path): _description_
    """
    gdf = gpd.read_file(geojson_path)
    assert gdf.crs is not None, "{} is missing a CRS"


def check_all_has_crs(paths: list[Path], workers: int, in_notebook: bool):
    """
    Parallelize has_crs over a list of Path objects.
    Errors out on the first failure.
    """
    this_tqdm = get_tqdm(use_async=False, in_notebook=in_notebook)
    # use fork instead of spawn to avoid semaphore leaks on macOS
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers) as pool:
        # executor.map will raise the first exception it encounters
        for _ in this_tqdm(
            pool.imap_unordered(has_crs, paths, chunksize=100),
            total=len(paths),
            desc="Checking CRS",
        ):
            pass
    logger.info(f"✅ All {len(paths)} files have a CRS.")


def is_within_n_hours(target_date: datetime, date_list: Iterable[datetime], n_hours: int) -> bool:
    """
    Returns True if target_date is within n_hours of any date in date_list.

    Args:
        target_date (datetime): The date to compare.
        date_list (list of datetime): List of other dates.
        n_hours (int): Number of hours as threshold.

    Returns:
        bool: True if within n_hours of any date in the list.
    """
    return any(abs(target_date - dt) <= timedelta(hours=n_hours) for dt in date_list)


def broad_band(all_bands: np.ndarray, no_data: np.ndarray) -> np.ndarray:
    # Natural colour broad band, log scaled
    #     red_recipe = 0.16666 * all_bands[5] + 0.66666 * all_bands[5] \
    #                  + 0.08333 * all_bands[5] + 0.4 * all_bands[6] + 0.4 * all_bands[7]
    #     green_recipe = 0.16666 *  all_bands[2] + 0.66666 *  all_bands[3] \
    #                    + 0.16666 *  all_bands[4]
    #     blue_recipe = 0.16666 *  all_bands[0] + 0.66666 *  all_bands[0] \
    #                    + 0.16666 *  all_bands[1]

    red_recipe = np.mean(all_bands[5:], axis=0)
    green_recipe = np.mean(all_bands[2:5], axis=0)
    blue_recipe = np.mean(all_bands[:2], axis=0)

    rgb_log = np.dstack((np.log10(1.0 + red_recipe), np.log10(1.0 + green_recipe), np.log10(1.0 + blue_recipe)))

    mins = np.array([rgb_log[:, :, i][~no_data].min() for i in range(3)])

    rgb_log -= mins
    rgb_log /= rgb_log.max(axis=(0, 1))

    rgb_log[no_data] = 0.0

    return rgb_log
