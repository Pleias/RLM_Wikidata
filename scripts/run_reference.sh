#!/usr/bin/env bash
# Run the evaluation (tasks/eval/, 100 tasks) with the reference configuration,
# then write the score, summary and timing reports into the batch directory.
#
# Prerequisites: a running endpoint (scripts/serve/serve_vllm.sh, or any
#   OpenAI-compatible server that serves $MODEL_NAME), the graph under
#   $WIKIDATA_GRAPH_DIR, and the harness environment (docs/setup.md).
#
# Usage:
#   bash scripts/run_reference.sh                     # the 100 evaluation tasks
#   TAG=smoke bash scripts/run_reference.sh --limit 1 # first task only
#   SKIP_PROBE=1 bash scripts/run_reference.sh        # no protocol probe
#
# Extra arguments go to scripts/bench/run_batch.py.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../env.sh
source "$REPO_ROOT/env.sh"
cd "$REPO_ROOT"

TAG="${TAG:-reference}"
BATCH="$REPO_ROOT/results/runs/$(date +%Y%m%d-%H%M)_${TAG}"

for required in meta.json nodes.parquet edges_fwd.arrow edges_bwd.arrow \
                properties.json entities_index.json label_index; do
    if [[ ! -e "$WIKIDATA_GRAPH_DIR/$required" ]]; then
        echo "missing $WIKIDATA_GRAPH_DIR/$required (see docs/setup.md)" >&2
        exit 1
    fi
done

bash scripts/serve/wait_for_server.sh "$BASE_URL" "$MODEL_NAME"

mkdir -p "$BATCH"
if [[ "${SKIP_PROBE:-0}" != "1" ]]; then
    # Hard gates: served model name, sub-query endpoint, stop handling, and
    # one usable Python cell per turn. A failure stops the run here.
    "$PYTHON" scripts/bench/probe_endpoint.py --skip-effort \
        --effort "$RLM_REASONING_EFFORT" --report "$BATCH/probe.json"
fi

"$PYTHON" scripts/bench/run_batch.py \
    --families ${FAMILIES:-eval} \
    --workers "${WORKERS:-1}" \
    --output-dir "$BATCH" \
    "$@"

"$PYTHON" scripts/bench/summary.py "$BATCH"            | tee "$BATCH/report_summary.txt"
"$PYTHON" scripts/bench/score.py "$BATCH"/*/          > "$BATCH/report_scores.txt"
"$PYTHON" scripts/bench/timing.py "$BATCH"            > "$BATCH/report_timing.txt"

echo
echo "batch:   $BATCH"
echo "reports: $BATCH/report_*.txt"
