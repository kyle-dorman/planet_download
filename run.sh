#!/bin/bash

# Exit immediately if any command fails
set -e

# Function to display usage instructions
usage() {
  echo "Usage: $0 <config_file> <years_comma_separated> <months_comma_separated>"
  echo "Example: $0 config.yaml 2022,2023 01,02,03"
  exit 1
}

# Check if exactly 3 arguments are provided
if [ "$#" -ne 3 ]; then
  usage
fi

# Assign input arguments
CONFIG_FILE="$1"
YEARS="$2"
MONTHS="$3"

# Validate that the config file exists
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: Config file '$CONFIG_FILE' does not exist."
  exit 1
fi

# Convert comma-separated years and months into arrays
IFS=',' read -r -a YEAR_LIST <<< "$YEARS"
IFS=',' read -r -a MONTH_LIST <<< "$MONTHS"

# Validate years (4-digit numbers)
for YEAR in "${YEAR_LIST[@]}"; do
  if ! [[ "$YEAR" =~ ^[0-9]{4}$ ]]; then
    echo "Error: Invalid year '$YEAR'. Year must be a 4-digit number."
    exit 1
  fi
done

# Validate months (01 to 12)
for MONTH in "${MONTH_LIST[@]}"; do
  if ! [[ "$MONTH" =~ ^(0[1-9]|1[0-2])$ ]]; then
    echo "Error: Invalid month '$MONTH'. Month must be between 01 and 12."
    exit 1
  fi
done

# Loop through each year and month and get all UDMs
for year in "${YEARS[@]}"
do
  for month in "${MONTHS[@]}"
  do    
    python src/download_udms.py --year $year --month $month --config-file $CONFIG_PATH
    
  done
done

# Loop through each year and month and select the best UDMs.
for year in "${YEARS[@]}"
do
  for month in "${MONTHS[@]}"
  do    
    python src/select_udms.py --year $year --month $month --config-file $CONFIG_PATH
    
  done
done

# Loop through each year and month and order images.
for year in "${YEARS[@]}"
do
  for month in "${MONTHS[@]}"
  do    
    python src/order_images.py --year $year --month $month --config-file $CONFIG_PATH
    
  done
done

# Loop through each year and month and unzip the contents.
for year in "${YEARS[@]}"
do
  for month in "${MONTHS[@]}"
  do    
    python src/unzip_downloads.py --year $year --month $month --config-file $CONFIG_PATH
    
  done
done

echo "All tasks completed!"
