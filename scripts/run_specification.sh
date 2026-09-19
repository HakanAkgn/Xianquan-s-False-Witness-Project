#!/usr/bin/env bash
# Execute in order, retaining baseline comparison before extensions.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/run_specification.sh NEW_EXECUTION_DIRECTORY [WORKERS]" >&2
  exit 2
fi
ROOT="$1"
WORKERS="${2:-4}"
if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite existing execution: $ROOT" >&2
  exit 2
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m unittest discover -s tests_replication -v
mkdir -p "$ROOT"
python -m replication.run --root "$ROOT/primary" --group primary --workers "$WORKERS"
python -m replication.run --root "$ROOT/access" --group access --workers "$WORKERS"
python -m replication.mechanism --primary "$ROOT/primary" --output "$ROOT/mechanism" --workers "$WORKERS"
python -m replication.assess --execution "$ROOT" --output "$ROOT/assessment"
python -m replication.extensions --primary "$ROOT/primary" --output "$ROOT/extensions" \
  --assessment "$ROOT/assessment/baseline_assessment.json" --workers "$WORKERS"
python -m replication.repair --primary "$ROOT/primary" --extensions "$ROOT/extensions" \
  --output "$ROOT/repair" --workers "$WORKERS"
python -m replication.audit --execution "$ROOT" --output "$ROOT/audit" --workers "$WORKERS"
python -m replication.summarize --execution "$ROOT" --output "$ROOT/summary"
