import sys
import os
import unittest
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predictor import predict_multi_horizon

class TestBTCPredictor(unittest.TestCase):
    def test_multi_horizon_prediction(self):
        """다양한 날짜(1일, 7일, 30일 등)의 예측값이 정상적으로 생성되는지 테스트합니다."""
        try:
            pred_df, current_price, actual_date = predict_multi_horizon()
            
            # 1. 예측 데이터프레임이 비어있지 않은지
            self.assertFalse(pred_df.empty, "예측 결과가 비어있습니다.")
            
            # 2. 필수 컬럼이 포함되어 있는지
            required_cols = ['horizon_days', 'target_date', 'predicted_price', 'predicted_pct_return']
            for col in required_cols:
                self.assertIn(col, pred_df.columns, f"필수 컬럼 {col}이(가) 없습니다.")
            
            # 3. 현재 가격이 유효한 숫자인지
            self.assertGreater(current_price, 0, "현재 가격은 0보다 커야 합니다.")
            
            print(f"✅ 테스트 성공: {len(pred_df)}개의 기간에 대한 예측값 생성 완료 (현재가: ${current_price:,.0f})")
            
        except Exception as e:
            self.fail(f"❌ 예측 중 오류 발생: {e}")

if __name__ == "__main__":
    unittest.main()
