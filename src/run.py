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

    # List of scripts to run
    scripts = ["src/download_udms.py", "src/select_udms.py", "src/order_images.py"]

    # Loop through scripts, years, and months
    for script in scripts:
        for y in year:
            for m in month:
                click.secho(f"🚀 Running {script} for Year: {y}, Month: {m}", fg="cyan")
                run_script(script, y, m, config_file)
                click.secho(f"✅ Finished Running {script} for Year: {y}, Month: {m}", fg="yellow")

    click.secho("✅ All tasks completed successfully!", fg="green")


if __name__ == "__main__":
    main()
