#!/bin/bash
# ==============================================================================
# BTC MLOps Pipeline Runner
# ==============================================================================
# This script is designed to be run by cron periodically (every 3 hours).
# It ensures the correct directory is used and logs all output.

# 1. Set Environment
# Ensure we have the path to standard binaries and user pip installs
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/Library/Python/3.9/bin:$PATH"

# 2. Navigate to Project Directory
# Resolve project root from this script location.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

# 3. Define Log File
LOG_FILE="$PROJECT_DIR/data/logs/cron_job.log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "============================================================" >> "$LOG_FILE"
echo "Starting Scheduled Pipeline Run: $(date)" >> "$LOG_FILE"
echo "Project Directory: $PROJECT_DIR" >> "$LOG_FILE"
echo "Working Directory: $(pwd)" >> "$LOG_FILE"
echo "============================================================" >> "$LOG_FILE"

# 4. Run Pipeline
# Policy: every 3 hours, always run full pipeline (data refresh + retrain).
# Ignore any external args to prevent accidental monitor-only/skip-train runs.
if [ "$#" -gt 0 ]; then
    echo "⚠️ Args ignored by policy. Forcing full pipeline run (data + retrain)." >> "$LOG_FILE"
fi
/usr/bin/python3 src/run_pipeline.py >> "$LOG_FILE" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Step TF: Transformer retrain sync" >> "$LOG_FILE"
    # Default behavior: train all positive horizons from config.
    # Optional override:
    #   TRANSFORMER_HORIZONS="1,2,3,5" ./scripts/run_daily.sh
    if [ -n "${TRANSFORMER_HORIZONS:-}" ]; then
        /usr/bin/python3 src/train_transformer.py --horizons "$TRANSFORMER_HORIZONS" >> "$LOG_FILE" 2>&1
    else
        /usr/bin/python3 src/train_transformer.py >> "$LOG_FILE" 2>&1
    fi
    TF_EXIT_CODE=$?
    if [ $TF_EXIT_CODE -ne 0 ]; then
        echo "⚠️ Pipeline succeeded, but Transformer retrain failed (exit $TF_EXIT_CODE) at $(date)" >> "$LOG_FILE"
    fi

    echo "Step RC: Reality Check sync" >> "$LOG_FILE"
    /usr/bin/python3 src/verify_reliability.py >> "$LOG_FILE" 2>&1
    RC_EXIT_CODE=$?
    if [ $TF_EXIT_CODE -eq 0 ] && [ $RC_EXIT_CODE -eq 0 ]; then
        echo "✅ Scheduled run completed successfully at $(date)" >> "$LOG_FILE"
    elif [ $TF_EXIT_CODE -ne 0 ] && [ $RC_EXIT_CODE -eq 0 ]; then
        echo "⚠️ Pipeline succeeded and Reality Check synced, but Transformer retrain failed at $(date)" >> "$LOG_FILE"
    else
        echo "⚠️ Pipeline succeeded, but Reality Check sync failed (exit $RC_EXIT_CODE) at $(date)" >> "$LOG_FILE"
    fi
else
    echo "❌ Scheduled run FAILED with exit code $EXIT_CODE at $(date)" >> "$LOG_FILE"
fi

echo "" >> "$LOG_FILE"
