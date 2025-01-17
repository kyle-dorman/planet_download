import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from src.config import DownloadConfig

logger = logging.getLogger(__name__)


# Get the type of Planet asset based on the date.
# 8 band wasn't available before 2021 (I think)
def asset_type_by_date(startdate: datetime) -> str:
    # for years before 2021, include 4-band imagery
    if startdate.year <= 2020:
        return "ortho_analytic_4b_sr"
    # for 2020 and years after, include 8-band imagery
    else:
        return "ortho_analytic_8b_sr"


# Get the type of Planet product bundle based on the date.
# 8 band wasn't available before 2021 (I think)
def product_bundle_by_date(startdate: datetime) -> str:
    # for years before 2021, include 4-band imagery
    if startdate.year <= 2020:
        return "analytic_sr_udm2"

    # for 2020 and years after, include 8-band imagery
    else:
        return "analytic_8b_sr_udm2"


def tif_paths(directory: Path) -> list[Path]:
    return sorted([pth for pth in directory.iterdir() if pth.suffix == ".tif"])


def geojson_paths(directory: Path) -> list[Path]:
    return sorted([pth for pth in directory.iterdir() if pth.suffix == ".geojson"])


# strip the _3B_udm2 from the file name
# e.g. 20230901_182511_53_2486_3B_udm2.tif
def cleaned_asset_id(udm_asset_id: str) -> str:
    return "_".join(udm_asset_id.split("_")[:4])


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


def create_config(config_file: Path, year: int, month: int) -> tuple[DownloadConfig, Path]:
    base_config = OmegaConf.structured(DownloadConfig)
    override_config = OmegaConf.load(config_file)
    config: DownloadConfig = OmegaConf.merge(base_config, override_config)  # type: ignore

    assert config.grid_dir.exists(), f"grid_dir {config.grid_dir} does not exist!"

    save_path = config.save_dir / str(year) / str(month).zfill(2)
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
