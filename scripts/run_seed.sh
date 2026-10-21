#!/usr/bin/env bash
# Usage: bash scripts/run_seed.sh <config.yaml> <seed> [extra tdi.train args]
# Runs training, probing and interventions for one seed. Prepare the data first:
#   python -m tdi.prepare_data --config <config.yaml>
set -euo pipefail
CONFIG=$1
SEED=$2
shift 2
NAME=$(python -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG'))['name'])")
RUN="runs/${NAME}_seed${SEED}"
python -m tdi.train --config "$CONFIG" --seed "$SEED" "$@"
python -m tdi.probing.run --run "$RUN"
python -m tdi.interventions.run --run "$RUN"
