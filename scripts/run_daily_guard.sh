#!/bin/bash
# ==============================================================================
# run_daily_guard.sh
# LaunchAgent에 의해 매시간 호출됨.
# 오늘 학습이 이미 완료됐으면 조용히 종료, 아직이면 run_daily.sh 실행.
# Mac이 자다 깨어나도 당일 학습을 반드시 한 번 보장.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

STAMP_FILE="$PROJECT_DIR/data/logs/last_trained_date.txt"
TODAY=$(date +%Y-%m-%d)

# 오늘 이미 실행했으면 스킵
if [ -f "$STAMP_FILE" ] && [ "$(cat "$STAMP_FILE")" = "$TODAY" ]; then
    exit 0
fi

# 자정~오전9시 사이는 대기 (너무 이른 시간 방지)
HOUR=$(date +%H)
if [ "$HOUR" -lt 9 ]; then
    exit 0
fi

# 학습 실행
bash "$SCRIPT_DIR/run_daily.sh"
EXIT_CODE=$?

# 성공 시 스탬프 기록
if [ $EXIT_CODE -eq 0 ]; then
    echo "$TODAY" > "$STAMP_FILE"
fi

exit $EXIT_CODE
