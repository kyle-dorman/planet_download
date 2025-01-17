import subprocess
import sys
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

from src.util import check_and_create_env


# Helper function to run each script
def run_script(script_path: str, year: int, month: int, config_file: Path) -> None:
    try:
        subprocess.run(
            ["python", script_path, "--year", str(year), "--month", str(month), "--config-file", str(config_file)],
            check=True,
        )
    except subprocess.CalledProcessError:
        click.secho(f"❌ Error: Failed to run {script_path} for Year: {year}, Month: {month}", fg="red")
        sys.exit(1)


@click.command()
@click.option("-c", "--config-file", type=click.Path(exists=True), required=True)
@click.option("-y", "--year", type=click.IntRange(min=1990, max=2050), multiple=True)
@click.option("-m", "--month", type=click.IntRange(min=1, max=12), multiple=True)
def main(config_file: Path, year: list[int], month: list[int]) -> None:
    """
    📥 Plant Downlowd Pipeline

    \b
    Arguments:
      CONFIG_FILE  - Path to the configuration file.
      YEAR        - List of years (e.g., --year 2022 --year 2023).
      MONTH       - List of months (e.g., --month 1 --month 2 --month 3).
    """
    config_file = Path(config_file)

    assert len(year) >= 1, "Must include at least 1 year"
    assert len(month) >= 1, "Must include at least 1 month"

    # Set the PlanetAPI Key in .env file if not set
    check_and_create_env()

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv(raise_error_if_not_found=True))

    # List of scripts to run
    scripts = ["src/scripts/download_udms.py", "src/scripts/select_udms.py", "src/scripts/order_images.py"]

    # Loop through scripts, years, and months
    for script in scripts:
        for y in year:
            for m in month:
                click.secho(f"🚀 Running {script} for Year: {y}, Month: {m}", fg="cyan")
                run_script(script_path=script, year=y, month=m, config_file=config_file)
                click.secho(f"Finished Running {script} for Year: {y}, Month: {m}", fg="green")

    click.secho("All tasks completed successfully!", fg="green")


if __name__ == "__main__":
    main()
