from dataclasses import dataclass
from pathlib import Path


@dataclass
class DownloadConfig:
    # Path the directory of grid geojson files
    grid_dir: Path = Path("/updateme")
    # Path to save the results (should not include a month/year)
    save_dir: Path = Path("/updateme")

    # The type of scene
    item_type: str = "PSScene"

    # Name of UDM asset
    udm_asset_type: str = "ortho_udm2"

    # Base name for Planet UDM search requests
    udm_search_name: str = "udm2_search"

    # Require ground control points
    ground_control: bool = True

    # Max allowed cloud cover percent
    cloud_cover: float = 1.0

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
