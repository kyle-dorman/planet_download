import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any


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
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

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
