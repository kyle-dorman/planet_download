from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ItemType(Enum):
    PSScene = "PSScene"
    SkySatCollect = "SkySatCollect"


class AssetType(Enum):
    ortho_sr = "ortho_sr"
    ortho = "ortho"
    basic = "basic"
    ortho_pansharpened = "ortho_pansharpened"


@dataclass
class DownloadConfig:
    # Path the directory of grid geojson files
    grid_dir: Path = Path("/updateme")

    # Path to save the results (should not include a month/year)

    save_dir: Path = Path("/updateme")

    # Path to copy the final surface reflectance and UDM data to.
    processing_dir: Path | None = None

    # Cleanup zip directory and full size UDMs
    cleanup: bool = True

    # The type of scene
    item_type: ItemType = ItemType.PSScene

    # Asset Type
    asset_type: AssetType = AssetType.ortho_sr

    # Number of bands to use
    num_bands: int = 8

    # Base name for Planet UDM search requests
    udm_search_name: str = "udm2_search"

    # Require ground control points
    ground_control: bool = True

    # Quality level
    quality_category: str = "standard"

    # Min allowed clear percent
    clear_percent: float = 0.0

    # Stage of imagegry data
    publishing_stage: str = "finalized"

    # Max number of UDMs to consider (for a single month ~60 is normal per grid)
    udm_limit: int = 1000

    # Max number of items in an order (will break a single order into multiple)
    order_item_limit: int = 500

    # Desired number of pixels per grid point across all images downloaded
    coverage_count: int = 5

    # Minimum percent of area an image needs to contribute to coverage to be considered
    percent_added: float = 0.05

    # Number of times to retry downloading imagegry data
    download_retries_max: int = 3

    # Seconds to wait before retrying
    download_backoff: float = 1.0

    # Only include one image per n days. Defer to use_same_range_if_neccessary otherwise.
    skip_same_range: int = 1

    # If there is less coverage than coverage_count use same date range items.
    use_same_range_if_neccessary: bool = True

    # Max grids to search at once
    grid_search_batch_size: int = 1000

    # Tides model directory
    tides_model_directory: Path | None = None

    # Tide model
    tide_model_name = "GOT4.10"

    # Tide model format
    tide_model_format = "GOT"


def validate_config(config: DownloadConfig):
    """Validate the DownloadConfig is compatable with planet api.

    Args:
        config (DownloadConfig): The config

    Raises:
        RuntimeError: If it is invalid.
    """
    # Verify we can create a valid udm string
    _ = udm_asset_string(config)

    # Verify we can create a valid asset string
    _ = planet_asset_string(config)

    # Verify we can create a valid product bundle
    _ = product_bundle_string(config)


def udm_asset_string(config: DownloadConfig) -> str:
    """Convert the config AssetType and ItemType to a udm string

    Args:
        config (DownloadConfig): The run config

    Raises:
        RuntimeError: For invalid AssetType & ItemType combinations

    Returns:
        str: The udm asset name string
    """
    if config.item_type == ItemType.PSScene:
        if config.asset_type == AssetType.basic:
            return "basic_udm2"
        elif config.asset_type in [AssetType.ortho, AssetType.ortho_sr]:
            return "ortho_udm2"
        else:
            raise RuntimeError(f"Unexpected AssetType {config.asset_type}")
    elif config.item_type == ItemType.SkySatCollect:
        if config.asset_type in [AssetType.ortho, AssetType.ortho_sr]:
            return "ortho_analytic_udm2"
        elif config.asset_type == AssetType.ortho_pansharpened:
            return "ortho_pansharpened_udm2"
        else:
            raise RuntimeError(f"Unexpected AssetType {config.asset_type}")
    else:
        raise RuntimeError(f"Unexpected ItemType {config.item_type}")


def planet_asset_string(config: DownloadConfig) -> str:
    if config.item_type == ItemType.PSScene:
        if config.asset_type == AssetType.ortho_sr:
            return f"ortho_analytic_{config.num_bands}b_sr"
        elif config.asset_type == AssetType.ortho:
            return f"ortho_analytic_{config.num_bands}b"
        elif config.asset_type == AssetType.basic:
            return f"basic_analytic_{config.num_bands}b"
        else:
            raise RuntimeError(f"Unexpected AssetType {config.asset_type}")
    elif config.item_type == ItemType.SkySatCollect:
        if config.asset_type == AssetType.ortho_sr:
            return "ortho_analytic_sr"
        elif config.asset_type == AssetType.ortho:
            return "ortho_analytic"
        elif config.asset_type == AssetType.ortho_pansharpened:
            return "ortho_pansharpened"
        else:
            raise RuntimeError(f"Unexpected AssetType {config.asset_type}")
    else:
        raise RuntimeError(f"Unexpected ItemType {config.item_type}")


def product_bundle_string(config: DownloadConfig) -> str:
    # Based on reference sheet:
    # https://developers.planet.com/apis/orders/product-bundles-reference/

    if config.item_type == ItemType.PSScene:
        if config.asset_type == AssetType.ortho_sr:
            if config.num_bands == 4:
                return "analytic_sr_udm2"
            elif config.num_bands == 8:
                return "analytic_8b_sr_udm2"
            else:
                raise RuntimeError(
                    "Unexpected number of bands {config.num_bands} for {config.item_type} and {config.asset_type}. Expected one of [4, 8]"
                )
        elif config.asset_type == AssetType.ortho:
            if config.num_bands == 3:
                return "analytic_3b_udm2"
            elif config.num_bands == 4:
                return "analytic_udm2"
            elif config.num_bands == 8:
                return "analytic_5b"
            else:
                raise RuntimeError(
                    "Unexpected number of bands {config.num_bands} for {config.item_type} and {config.asset_type}. Expected one of [3, 4, 8]"
                )
        elif config.asset_type == AssetType.basic:
            if config.num_bands == 4:
                return "basic_analytic_udm2"
            elif config.num_bands == 8:
                return "basic_analytic_8b_udm2"
            else:
                raise RuntimeError(
                    "Unexpected number of bands {config.num_bands} for {config.item_type} and {config.asset_type}. Expected one of [4, 8]"
                )
        else:
            raise RuntimeError(f"Unexpected AssetType {config.asset_type}")
    elif config.item_type == ItemType.SkySatCollect:
        if config.asset_type in [AssetType.ortho, AssetType.ortho_sr]:
            return "analytic_udm2"
        elif config.asset_type == AssetType.ortho_pansharpened:
            return "pansharpened_udm2"
        else:
            raise RuntimeError(f"Unexpected AssetType {config.asset_type}")
    else:
        raise RuntimeError(f"Unexpected ItemType {config.item_type}")
