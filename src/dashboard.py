import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go

# 프로젝트 루트를 경로에 추가 (설정 파일을 불러오기 위함)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import MODELS_DIR, PROCESSED_DIR
    from src.config import cfg
    from src.predictor import predict_multi_horizon
except ImportError:
    st.error("❌ 필수 모듈을 불러오지 못했습니다. 경로 설정을 확인하세요.")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# 🌈 무지개 스타일 정의 (사용자 요청 반영)
# ──────────────────────────────────────────────────────────────────────────────
def apply_custom_style():
    """중학생도 이해할 수 있는 화려한 무지개 스타일과 이중 물결 배경을 적용합니다!"""
    st.markdown("""
    <style>
    /* 1. 배경을 전체적으로 어둡게 하고, 이중 물결 애니메이션 적용 */
    .stApp {
        background-color: #0e1117;
    }

    /* 뒤쪽 레이어: 천천히 흐르는 메인 색상 배경 */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        z-index: -2;
        background: linear-gradient(135deg, #0e1117 0%, #1a2235 50%, #0e1117 100%);
        background-size: 200% 200%;
        animation: wave-bg 12s ease-in-out infinite alternate;
    }

    /* 앞쪽 레이어: 물결이 일렁이는 느낌을 주는 부드러운 빛 효과 */
    .stApp::after {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        z-index: -1;
        background: 
            radial-gradient(circle at 30% 70%, rgba(30, 80, 150, 0.15) 0%, transparent 60%),
            radial-gradient(circle at 70% 30%, rgba(150, 60, 30, 0.1) 0%, transparent 60%);
        animation: wave-fluid 15s ease-in-out infinite alternate;
        pointer-events: none;
    }

    @keyframes wave-bg {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }

    @keyframes wave-fluid {
        0% { transform: scale(1) translateY(0px) rotate(0deg); }
        100% { transform: scale(1.1) translateY(-30px) rotate(5deg); }
    }

    /* 2. 무지개처럼 반짝이며 움직이는 애니메이션 정의 */
    @keyframes rainbow-move {
      0% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
    }

    /* 3. 무지개 텍스트 클래스 */
    .rainbow-active-text {
      background: linear-gradient(
        to right, 
        #ff0000, #ff7f00, #ffff00, #00ff00, #0000ff, #4b0082, #8b00ff
      );
      background-size: 200% auto;
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
      animation: rainbow-move 3s linear infinite;
      display: inline-block;
      font-weight: 800;
      font-size: 3.5rem;
      margin-bottom: 0.5rem;
      line-height: 1.2;
    }

    /* 4. 카드 스타일 */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #ff7f00;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 📊 메인 대시보드 함수
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="BTCn | 비트코인 정밀 예측",
        page_icon="🤖",
        layout="wide"
    )
    apply_custom_style()

    # 상단 헤더
    st.markdown('<h1 class="rainbow-active-text">BTCn: Bitcoin Forecast</h1>', unsafe_allow_html=True)
    st.markdown("### 인공지능이 분석한 비트코인의 미래 수익률")
    st.write("---")

    # 데이터 로드 및 예측 수행
    with st.spinner("AI 모델이 미래를 계산 중입니다..."):
        try:
            pred_df, current_price, actual_date = predict_multi_horizon()
        except Exception as e:
            st.error(f"⚠️ 예측 데이터를 불러오는 데 실패했습니다: {e}")
            st.info("터미널에서 `python -m src.run_pipeline`을 실행하여 모델을 먼저 학습시켜주세요.")
            return

    # 실시간 요약 (Metrics)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("현재 시세", f"${current_price:,.0f}", delta=None)
    with col2:
        st.metric("데이터 기준일", actual_date.strftime('%Y-%m-%d'))
    
    # 7일 후 예측
    p7 = pred_df[pred_df['horizon_days'] == 7]
    if not p7.empty:
        ret_7d = p7.iloc[0]['predicted_pct_return']
        with col3:
            st.metric("7일 후 예측", f"{ret_7d:+.1f}%", delta=f"{ret_7d:.1f}%")
            
    # 30일 후 예측
    p30 = pred_df[pred_df['horizon_days'] == 30]
    if not p30.empty:
        ret_30d = p30.iloc[0]['predicted_pct_return']
        with col4:
            st.metric("30일 후 예측", f"{ret_30d:+.1f}%", delta=f"{ret_30d:.1f}%")

    st.write("")
    
    # 메인 그래프 및 상세 표
    tab1, tab2 = st.tabs(["📈 가격 전망 차트", "📋 상세 예측 데이터"])

    with tab1:
        st.subheader("🔮 향후 1년 가격 프로젝션")
        fig = go.Figure()
        
        # 현재 시점 표시
        fig.add_trace(go.Scatter(
            x=[actual_date], 
            y=[current_price], 
            mode='markers+text', 
            name='현재 시점',
            text=["현재"],
            textposition="top center",
            marker=dict(size=15, color='#00ff00', symbol='diamond')
        ))
        
        # 미래 예측 선
        fig.add_trace(go.Scatter(
            x=pred_df['target_date'], 
            y=pred_df['predicted_price'], 
            mode='lines+markers', 
            name='AI 예측 경로',
            line=dict(color='#ff7f00', width=4, dash='solid'),
            marker=dict(size=8, color='#ff7f00'),
            hovertemplate="날짜: %{x}<br>예측가: $%{y:,.0f}<extra></extra>"
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=600,
            xaxis_title="날짜 (Time)",
            yaxis_title="비트코인 가격 (USD)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("예측 데이터 요약")
        # 보기 좋게 가공
        tdf = pred_df[['horizon_days', 'target_date', 'predicted_price', 'predicted_pct_return', 'model_name']].copy()
        tdf.columns = ['예측 기간(일)', '목표 날짜', '예측 가격($)', '예상 수익률(%)', '사용 모델']
        tdf['목표 날짜'] = tdf['목표 날짜'].dt.date
        
        st.dataframe(
            tdf.sort_values("예측 기간(일)").style.format({
                '예측 가격($)': '{:,.0f}',
                '예상 수익률(%)': '{:+.2f}%'
            }), 
            use_container_width=True
        )

    # 하단 정보
    st.write("---")
    st.caption(f"🤖 BTCn 머신러닝 시스템 | 마지막 동기화: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
