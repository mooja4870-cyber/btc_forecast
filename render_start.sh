#!/bin/bash
# Render Start Command 스크립트
# 1) Google Drive에서 모델 다운로드
# 2) Streamlit 앱 실행

echo "========================================="
echo "  BTC Prediction App — Starting..."
echo "========================================="

# Step 1: 모델 다운로드
echo "🔄 Step 1: 모델 파일 확인 및 다운로드..."
python download_models.py

# Step 2: Streamlit 실행
echo "🚀 Step 2: Streamlit 앱 시작..."
streamlit run app.py \
    --server.port "$PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
