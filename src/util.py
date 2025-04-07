import asyncio
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Type

from matplotlib.dates import relativedelta
from omegaconf import OmegaConf
from tqdm.asyncio import tqdm_asyncio
from tqdm.notebook import tqdm_notebook
from tqdm.std import tqdm

from src.config import DownloadConfig, validate_config

logger = logging.getLogger(__name__)


def match_tif_path(filepath: Path) -> re.Match[str]:
    #: Regular expression used to extract info from planet filename.
    # <acquisition date>_<acquisition time>_<satellite_id>_<productLevel>_<bandProduct>.<extension>
    # 20221001_175121_27_2460_3B_udm2.tif
    filename_regex = r"""
    (?P<date>\d{8}_\d{6}_\d{2})_     # Acquisition date (YYYYMMDD_HHMMSS_xx)
    (?P<satellite_id>\w{4})_         # Satellite ID (4 characters)
    (?P<product_level>\w{2})_        # Product level (2 digits)
    (?P<band_product>[\w]+)          # Band product (letters or digits)
    \.(?P<extension>\w+)             # Rest of the file info (no periods)
    """

    filename_regex = re.compile(filename_regex, re.VERBOSE)
    match = re.match(filename_regex, filepath.name)
    if match is None:
        raise RuntimeError(f"Could not parse tif filename {filepath.name}")

    return match


def match_tif_path2(filepath: Path) -> re.Match[str]:
    #: Regular expression used to extract info from planet filename.
    # <acquisition date>_<acquisition time>_<satellite_id>_<productLevel>_<bandProduct>.<extension>
    # 20190312_181227_103a_3B_udm2.tif
    filename_regex = r"""
    (?P<date>\d{8}_\d{6})_           # Acquisition date (YYYYMMDD_HHMMSS)
    (?P<satellite_id>\w{4})_         # Satellite ID (4 characters)
    (?P<product_level>\w{2})_        # Product level (2 digits)
    (?P<band_product>[\w]+)          # Band product (letters or digits)
    \.(?P<extension>\w+)             # Rest of the file info (no periods)
    """

    filename_regex = re.compile(filename_regex, re.VERBOSE)
    match = re.match(filename_regex, filepath.name)
    if match is None:
        raise RuntimeError(f"Could not parse tif filename {filepath.name}")

    return match


def match_tif_path3(filepath: Path) -> re.Match[str]:
    #: Regular expression used to extract info from planet filename.
    # <acquisition date>_<acquisition time>_<satellite_id>_<productLevel>_<bandProduct>.<extension>
    # 20200223_163753_1_0f49_3B_udm2.tif
    filename_regex = r"""
    (?P<date>\d{8}_\d{6}_\d{1})_     # Acquisition date (YYYYMMDD_HHMMSS_x)
    (?P<satellite_id>\w{4})_         # Satellite ID (4 characters)
    (?P<product_level>\w{2})_        # Product level (2 digits)
    (?P<band_product>[\w]+)          # Band product (letters or digits)
    \.(?P<extension>\w+)             # Rest of the file info (no periods)
    """

    filename_regex = re.compile(filename_regex, re.VERBOSE)
    match = re.match(filename_regex, filepath.name)
    if match is None:
        raise RuntimeError(f"Could not parse tif filename {filepath.name}")

    return match


def parse_tif_path(filepath: Path) -> datetime:
    #: Date format string used to parse date from filename.
    try:
        match = match_tif_path(filepath)
        date_format = "%Y%m%d_%H%M%S_%f"
    except RuntimeError:
        try:
            match = match_tif_path2(filepath)
            date_format = "%Y%m%d_%H%M%S"
        except RuntimeError:
            match = match_tif_path3(filepath)
            date_format = "%Y%m%d_%H%M%S_%f"

    datestr = match.group("date")
    tif_datetime = datetime.strptime(datestr, date_format)

    return tif_datetime


def tif_paths(directory: Path) -> list[Path]:
    return sorted([pth for pth in directory.iterdir() if pth.suffix == ".tif"])


def geojson_paths(directory: Path) -> list[Path]:
    return sorted([pth for pth in directory.iterdir() if pth.suffix == ".geojson"])


# strip the _3B_udm2 from the file name
# e.g. 20230901_182511_53_2486_3B_udm2.tif
def cleaned_asset_id(filepath: Path) -> str:
    try:
        match = match_tif_path(filepath)
    except RuntimeError:
        try:
            match = match_tif_path2(filepath)
        except RuntimeError:
            match = match_tif_path3(filepath)

    date = match.group("date")
    satellite_id = match.group("satellite_id")

    return date + "_" + satellite_id


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
