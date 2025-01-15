import subprocess
import sys
from pathlib import Path

import click


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
@click.option("-y", "--years", type=click.IntRange(min=1990, max=2050), multiple=True)
@click.option("-m", "--months", type=click.IntRange(min=1, max=12), multiple=True)
def main(config_file: Path, years: list[int], months: list[int]) -> None:
    """
    📥 Plant Downlowd Pipeline

    \b
    Arguments:
      CONFIG_FILE  - Path to the configuration file.
      YEARS        - Space-separated list of years (e.g., 2022 2023).
      MONTHS       - Space-separated list of months (e.g., 1 2 3).
    """
    config_file = Path(config_file)
    assert len(years) == len(months), f"Must pass same number of years({len(years)}) and months({len(months)})!"

    # List of scripts to run
    scripts = ["src/download_udms.py", "src/select_udms.py", "src/order_images.py", "src/unzip_downloads.py"]

    # Loop through scripts, years, and months
    for script in scripts:
        for year in years:
            for month in months:
                click.secho(f"🚀 Running {script} for Year: {year}, Month: {month}", fg="cyan")
                run_script(script, year, month, config_file)
                click.secho(f"Finished Running {script} for Year: {year}, Month: {month}", fg="yellow")

    click.secho("✅ All tasks completed successfully!", fg="green")


if __name__ == "__main__":
    main()
