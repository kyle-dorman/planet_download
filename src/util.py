import logging
from datetime import datetime
from pathlib import Path


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


# Configure a logger. Add console and (optionally) file loggers.
def setup_logger(logger: logging.Logger, save_dir: Path | None = None, log_filename: str = "log.log"):
    logging.getLogger("planet").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Remove base handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set level for logger
    logger.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if save_dir is not None:
        # Create handler
        file_handler = logging.FileHandler(save_dir / log_filename)  # Logs to a file

        # Attach formatter to the handler
        file_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)

    # Create handler
    console_handler = logging.StreamHandler()  # Logs to console

    # Attach formatter to the handler
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
