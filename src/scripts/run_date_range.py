import subprocess
import sys
from datetime import datetime
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

from src.util import check_and_create_env


# Helper function to run each script
def run_script(script_path: str, start_date: datetime, end_date: datetime, config_file: Path) -> None:
    try:
        subprocess.run(
            [
                "python",
                script_path,
                "--start-date",
                start_date.strftime("%Y-%m-%d"),
                "--end-date",
                end_date.strftime("%Y-%m-%d"),
                "--config-file",
                str(config_file),
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        click.secho(
            f"❌ Error: Failed to run {script_path} for start-date: {start_date}, end-date: {end_date}", fg="red"
        )
        sys.exit(1)


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
):
    """
    📥 Plant Downlowd Pipeline

    \b
    Arguments:
      CONFIG_FILE  - Path to the configuration file.
      start_date - The start date
      end_date - The end date
    """
    config_file = Path(config_file)

    # Set the PlanetAPI Key in .env file if not set
    check_and_create_env()

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv(raise_error_if_not_found=True))

    # List of scripts to run
    scripts = [
        "src/scripts/udm_search.py",
        "src/scripts/udm_activate.py",
        "src/scripts/udm_download.py",
        "src/scripts/udm_select.py",
        "src/scripts/order_create.py",
        "src/scripts/order_download.py",
    ]

    # Loop through scripts, years, and months
    for script in scripts:
        click.secho(
            f"🚀 Running {script} for Start: {start_date.strftime('%Y-%m-%d')}, End: {end_date.strftime('%Y-%m-%d')}",
            fg="cyan",
        )

        run_script(script_path=script, start_date=start_date, end_date=end_date, config_file=config_file)

        click.secho(
            f"Finished Running {script} for Start: {start_date.strftime('%Y-%m-%d')}, End: {end_date.strftime('%Y-%m-%d')}",
            fg="green",
        )

        click.secho("All tasks completed successfully!", fg="green")


if __name__ == "__main__":
    main()
