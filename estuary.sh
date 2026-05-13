#!/bin/bash
set -euo pipefail  # Exit on error, undefined var, or pipe failure

BASE="/Volumes/x10pro/estuary_global/sat_data"

# scripts = [
#     "src/scripts/udm_search.py",
#     "src/scripts/udm_activate.py",
#     "src/scripts/udm_download.py",
#     "src/scripts/udm_select.py",
#     "src/scripts/udm_cleanup.py",
#     "src/scripts/order_create.py",
#     "src/scripts/order_download.py",
# ]

CMDS=(
  udm_select
)

for YEAR in $(seq 2017 2017); do
  START_DATE="${YEAR}-01-01"
  NEXT_YEAR=$((YEAR + 1))
  END_DATE="${NEXT_YEAR}-01-01"

  for SENSOR in superdove dove; do
    CONFIG_FILE="${BASE}/${SENSOR}/config.yaml"

    if [[ ! -f "$CONFIG_FILE" ]]; then
      echo "Missing config file: $CONFIG_FILE"
      exit 1
    fi

    echo "Running for ${SENSOR} ${YEAR}..."

    for cmd in "${CMDS[@]}"; do
      python src/scripts/${cmd}.py \
        --config-file "$CONFIG_FILE" \
        --start-date "$START_DATE" \
        --end-date "$END_DATE"
    done
  done
done
