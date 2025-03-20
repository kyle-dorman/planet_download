from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path


# Get the type of Planet asset based on the date.
# 8 band wasn't available before 2021 (I think)
def num_bands_from_datetime(capture_datetime: datetime) -> int:
    return 4 if capture_datetime.year <= 2020 else 8


class AssetType(Enum):
    ortho_sr = "ortho_sr"
    ortho = "ortho"
    basic = "basic"

    def udm_asset_string(self) -> str:
        if self == AssetType.basic:
            return "basic_udm2"
        elif self in [AssetType.ortho, AssetType.ortho_sr]:
            return "ortho_udm2"
        else:
            raise RuntimeError(f"Unexpected AssetType {self}")

    def planet_asset_string(self, capture_datetime: datetime) -> str:
        num_bands = num_bands_from_datetime(capture_datetime)

        if self == AssetType.ortho_sr:
            return f"ortho_analytic_{num_bands}b_sr"
        elif self == AssetType.ortho:
            return f"ortho_analytic_{num_bands}b"
        elif self == AssetType.basic:
            return f"basic_analytic_{num_bands}b"
        else:
            raise RuntimeError(f"Unexpected AssetType {self}")

    def product_bundle_string(self, capture_datetime: datetime) -> str:
        # Based on reference sheet:
        # https://developers.planet.com/apis/orders/product-bundles-reference/

        num_bands = num_bands_from_datetime(capture_datetime)
        if num_bands == 8:
            num_bands_str = "8b_"
        else:
            num_bands_str = ""

        if self == AssetType.ortho_sr:
            return f"analytic_{num_bands_str}sr_udm2"

        elif self == AssetType.ortho:
            # Crazy logic, untested
            if num_bands == 8:
                return "analytic_5b"
            else:
                return "analytic_udm2"

        elif self == AssetType.basic:
            return f"basic_analytic_{num_bands_str}udm2"
        else:
            raise RuntimeError(f"Unexpected AssetType {self}")


@dataclass
class DownloadConfig:
    # Path the directory of grid geojson files
    grid_dir: Path = Path("/updateme")

    # Path to save the results (should not include a month/year)

    save_dir: Path = Path("/updateme")

    # Path to copy the final surface reflectance and UDM data to.
    processing_dir: Path | None = None

    # The type of scene
    item_type: str = "PSScene"

    # Asset Type
    asset_type: AssetType = AssetType.ortho_sr

    # Base name for Planet UDM search requests
    udm_search_name: str = "udm2_search"

    # Require ground control points
    ground_control: bool = True

    # Min allowed clear percent
    clear_percent: float = 0.0

    # Stage of imagegry data
    publishing_stage: str = "finalized"

    # Max number of UDMs to consider (for a single month ~60 is normal per grid)
    udm_limit: int = 1000

    # Desired number of pixels per grid point across all images downloaded
    coverage_count: int = 5

    # Minimum percent of area an image needs to contribute to coverage to be considered
    percent_added: float = 0.05

    # The ground sample distance of the data (Can we get there somewhere else?)
    ground_sample_distance: float = 3.0

    # Number of times to retry downloading imagegry data
    download_retries_max: int = 3

    # Seconds to wait before retrying
    download_backoff: float = 1.0

    # Only include one image per day unless there are less than coverage_count and it is needed.
    skip_same_day: bool = True

    # Tides model directory
    tides_model_directory: Path | None = None

    # Tide model
    tide_model_name = "GOT4.10"

    # Tide model format
    tide_model_format = "GOT"
