#!/bin/bash
# ==============================================================================
# BTC MLOps Pipeline Runner
# ==============================================================================
# This script is designed to be run by cron once daily at 00:00.
# It ensures the correct directory is used and logs all output.

# 1. Set Environment
# Ensure we have the path to standard binaries and user pip installs
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/Library/Python/3.9/bin:$PATH"

# 2. Navigate to Project Directory
# Resolve project root from this script location.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

# Resolve Python executable with strict fallback order.
# Hard guard: only use interpreter that can import required runtime modules.
PYTHON_BIN=""
for CANDIDATE in \
    "${PROJECT_DIR}/.venv/bin/python3" \
    "${PROJECT_DIR}/venv/bin/python3" \
    "$(command -v python3 2>/dev/null)"
do
    if [ -n "$CANDIDATE" ] && [ -x "$CANDIDATE" ]; then
        if "$CANDIDATE" -c "import yaml, pandas, numpy" >/dev/null 2>&1; then
            PYTHON_BIN="$CANDIDATE"
            break
        fi
    fi
done

# 3. Define Log File
LOG_FILE="$PROJECT_DIR/data/logs/cron_job.log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "============================================================" >> "$LOG_FILE"
echo "Starting Scheduled Pipeline Run: $(date)" >> "$LOG_FILE"
echo "Project Directory: $PROJECT_DIR" >> "$LOG_FILE"
echo "Working Directory: $(pwd)" >> "$LOG_FILE"
echo "Python Executable: ${PYTHON_BIN:-NOT_FOUND}" >> "$LOG_FILE"
echo "============================================================" >> "$LOG_FILE"

if [ -z "$PYTHON_BIN" ]; then
    echo "❌ Scheduled run FAILED: usable python3 not found (missing required modules) at $(date)" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    exit 1
fi

# 4. Run Pipeline
# Policy: once daily at 00:00, always run full pipeline (data refresh + retrain).
# Ignore any external args to prevent accidental monitor-only/skip-train runs.
if [ "$#" -gt 0 ]; then
    echo "⚠️ Args ignored by policy. Forcing full pipeline run (data + retrain)." >> "$LOG_FILE"
fi
"$PYTHON_BIN" src/run_pipeline.py >> "$LOG_FILE" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Step TF: Transformer retrain sync" >> "$LOG_FILE"
    # Default behavior: train all positive horizons from config.
    # Optional override:
    #   TRANSFORMER_HORIZONS="1,2,3,5" ./scripts/run_daily.sh
    if [ -n "${TRANSFORMER_HORIZONS:-}" ]; then
        "$PYTHON_BIN" src/train_transformer.py --horizons "$TRANSFORMER_HORIZONS" >> "$LOG_FILE" 2>&1
    else
        "$PYTHON_BIN" src/train_transformer.py >> "$LOG_FILE" 2>&1
    fi
    TF_EXIT_CODE=$?
    if [ $TF_EXIT_CODE -ne 0 ]; then
        echo "⚠️ Pipeline succeeded, but Transformer retrain failed (exit $TF_EXIT_CODE) at $(date)" >> "$LOG_FILE"
    fi

    echo "Step RC: Reality Check sync" >> "$LOG_FILE"
    "$PYTHON_BIN" src/verify_reliability.py >> "$LOG_FILE" 2>&1
    RC_EXIT_CODE=$?
    if [ $TF_EXIT_CODE -eq 0 ] && [ $RC_EXIT_CODE -eq 0 ]; then
        echo "✅ Scheduled run completed successfully at $(date)" >> "$LOG_FILE"
    elif [ $TF_EXIT_CODE -ne 0 ] && [ $RC_EXIT_CODE -eq 0 ]; then
        echo "⚠️ Pipeline succeeded and Reality Check synced, but Transformer retrain failed at $(date)" >> "$LOG_FILE"
    else
        echo "⚠️ Pipeline succeeded, but Reality Check sync failed (exit $RC_EXIT_CODE) at $(date)" >> "$LOG_FILE"
    fi

    # 학습 완료 후 GitHub에 자동 push (Render 배포 앱 최신화)
    echo "Step GIT: Pushing updated data & models to GitHub..." >> "$LOG_FILE"
    cd "$PROJECT_DIR" || exit 1
    git config user.name "local-cron" 2>/dev/null
    git config user.email "local-cron@localhost" 2>/dev/null
    # data/·models/는 .gitignore 등재 자산이므로 -f로 강제 스테이징 (add 누락 시 exit 1 → 푸시 실패 방지)
    git add -f data/ models/ >> "$LOG_FILE" 2>&1
    GIT_COMMIT_MSG="Auto-update data, models, and logs: $(date +'%Y-%m-%d %H:%M KST')"
    if git diff --cached --quiet; then
        echo "Nothing to commit." >> "$LOG_FILE"
    else
        git commit -m "$GIT_COMMIT_MSG" >> "$LOG_FILE" 2>&1
    fi
    git pull --rebase --autostash origin main >> "$LOG_FILE" 2>&1
    git push origin main >> "$LOG_FILE" 2>&1
    GIT_EXIT=$?
    if [ $GIT_EXIT -eq 0 ]; then
        echo "✅ GitHub push 완료 at $(date)" >> "$LOG_FILE"
    else
        echo "⚠️ GitHub push 실패 (exit $GIT_EXIT) at $(date)" >> "$LOG_FILE"
    fi
else
    echo "❌ Scheduled run FAILED with exit code $EXIT_CODE at $(date)" >> "$LOG_FILE"
fi

echo "" >> "$LOG_FILE"
