#!/bin/bash
set -euo pipefail

# Lint

# Pyupgrade
find src -name "*.py" -print0 | xargs -0 pyupgrade

# isort
isort src

# black
black src

# flake
flake8 src
