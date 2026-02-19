"""
Monitoring Module
=================
Handles metric calculation, drift detection, and report generation.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging

from src.config import cfg

logger = logging.getLogger("monitoring")

class MetricCalculator:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Compute standard regression metrics."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if not mask.any():
            return {}
            
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Directional Accuracy (log returns)
        # Note: input is price, so we need previous price to calculate direction?
        # A simplified proxy check: sign(pred_price - prev_price) == sign(actual_price - prev_price)
        # For now, let's stick to error metrics as direction needs time-series context
        
        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "count": int(len(y_true))
        }

class DriftDetector:
    @staticmethod
    def check_feature_drift(current_df, reference_df, threshold_sigma=2.0):
        """
        Simple Z-score drift detection for feature means.
        Returns list of features with significant drift.
        """
        drifted_features = []
        # Find common numeric columns
        cols = [c for c in current_df.columns if c in reference_df.columns and pd.api.types.is_numeric_dtype(current_df[c])]
        
        for col in cols:
            ref_mean = reference_df[col].mean()
            ref_std = reference_df[col].std()
            curr_mean = current_df[col].mean()
            
            if ref_std == 0:
                continue
                
            z_score = abs(curr_mean - ref_mean) / ref_std
            if z_score > threshold_sigma:
                drifted_features.append({
                    "feature": col,
                    "z_score": float(z_score),
                    "current_mean": float(curr_mean),
                    "ref_mean": float(ref_mean)
                })
        
        return drifted_features

class ReportGenerator:
    def __init__(self, processed_dir, reports_dir):
        self.processed_dir = processed_dir
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)

    def _load_featured_dataset(self) -> pd.DataFrame:
        path = os.path.join(self.processed_dir, "featured_dataset.csv")
        if not os.path.exists(path):
            return pd.DataFrame()
        try:
            return pd.read_csv(path, index_col=0, parse_dates=True)
        except Exception as e:
            logger.warning(f"Failed to load featured dataset for monitoring: {e}")
            return pd.DataFrame()

    def _tracked_expansion_features(self, fdf: pd.DataFrame) -> list:
        if fdf.empty:
            return []
        tracked = []
        tracked += [c for c in fdf.columns if "_fut_" in c]
        tracked += [c for c in fdf.columns if c.startswith("rate_") or "policy_rate" in c or "fomc" in c or c.startswith("curve_")]
        tracked += [c for c in fdf.columns if c.startswith("geo_") or c.endswith("_intensity_score") or c.endswith("_shock_score")]
        # stable order + dedup
        return list(dict.fromkeys(tracked))

    def _feature_quality_snapshot(self, fdf: pd.DataFrame) -> dict:
        if fdf.empty:
            return {"tracked_count": 0, "rows": []}

        tracked = self._tracked_expansion_features(fdf)
        if not tracked:
            return {"tracked_count": 0, "rows": []}

        rows = []
        now_idx = fdf.index.max()
        for col in tracked:
            if col not in fdf.columns:
                continue
            series = fdf[col]
            recent = series.tail(30)
            missing_pct = float(recent.isna().mean() * 100.0)
            last_valid = series.last_valid_index()
            stale_days = int((now_idx - last_valid).days) if last_valid is not None else 10**9
            rows.append({
                "feature": col,
                "missing_pct_recent_30d": round(missing_pct, 2),
                "stale_days": stale_days,
            })
        rows = sorted(rows, key=lambda r: (-r["missing_pct_recent_30d"], -r["stale_days"], r["feature"]))
        return {"tracked_count": len(tracked), "rows": rows}

    def _expansion_feature_drift(self, fdf: pd.DataFrame) -> list:
        if fdf.empty or len(fdf) < 240:
            return []
        tracked = self._tracked_expansion_features(fdf)
        if not tracked:
            return []

        recent = fdf[tracked].tail(30)
        reference = fdf[tracked].iloc[-210:-30]
        if recent.empty or reference.empty:
            return []
        sigma = cfg.monitoring.get("drift", {}).get("feature_mean_shift_sigma", 2.0)
        return DriftDetector.check_feature_drift(recent, reference, threshold_sigma=sigma)

    def generate_report(self, eval_df, run_id="N/A"):
        """Generate a Markdown report from evaluation data."""
        if eval_df.empty:
            logger.warning("No evaluation data to generate report.")
            return

        report_path = os.path.join(self.reports_dir, "monitoring_report.md")
        
        # Calculate overall metrics
        overall = MetricCalculator.calculate_metrics(eval_df["actual_price"], eval_df["predicted_price"])
        
        # Calculate rolling 30d metrics
        recent_mask = eval_df["target_date"] >= (pd.Timestamp.now() - pd.Timedelta(days=30))
        recent_df = eval_df[recent_mask]
        recent = MetricCalculator.calculate_metrics(recent_df["actual_price"], recent_df["predicted_price"])
        
        # Alerts check
        alerts = []
        thresholds = cfg.monitoring.get("alerts", {})
        if recent.get("mape", 0) > thresholds.get("mape_threshold", 15.0):
            alerts.append(f"⚠️ Recent MAPE ({recent['mape']:.1f}%) exceeds threshold ({thresholds['mape_threshold']}%)")

        fdf = self._load_featured_dataset()
        quality = self._feature_quality_snapshot(fdf)
        drift = self._expansion_feature_drift(fdf)
        severe_quality = [r for r in quality.get("rows", []) if r["missing_pct_recent_30d"] >= 30.0 or r["stale_days"] >= 7]
        if severe_quality:
            alerts.append(f"⚠️ Expansion features quality issue count: {len(severe_quality)}")
        if drift:
            alerts.append(f"⚠️ Expansion feature drift detected: {len(drift)} features")
            
        md = f"""# 📊 BTC Model Monitoring Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Run ID:** {run_id}

## 🚨 Status Dashboard
{'✅ All Systems Normal' if not alerts else '❌ Alerts Active'}
{chr(10).join(['- ' + a for a in alerts])}

## 📈 Performance Metrics

| Metric | Overall (All Time) | Last 30 Days |
| :--- | :--- | :--- |
| **MAE** | ${overall.get('mae', 0):.2f} | ${recent.get('mae', 0):.2f} |
| **RMSE** | ${overall.get('rmse', 0):.2f} | ${recent.get('rmse', 0):.2f} |
| **MAPE** | {overall.get('mape', 0):.1f}% | {recent.get('mape', 0):.1f}% |
| **Count** | {overall.get('count', 0)} | {recent.get('count', 0)} |

## 📉 Recent Error Trend
*(Last 5 predictions)*
"""
        # Recent predictions table
        tail = eval_df.sort_values("target_date").tail(5)
        md += tail[["target_date", "horizon", "predicted_price", "actual_price", "error_pct"]].to_markdown(index=False)

        md += "\n\n## 🧩 Expansion Feature Health\n"
        md += f"- Tracked features: {quality.get('tracked_count', 0)}\n"
        md += f"- Drifted features (30d vs prev 180d): {len(drift)}\n"

        if quality.get("rows"):
            qdf = pd.DataFrame(quality["rows"]).head(15)
            md += "\n### Quality Snapshot (Top 15 by missing/staleness)\n"
            md += qdf.to_markdown(index=False)
        else:
            md += "\n_No expansion features currently active or available._\n"

        if drift:
            ddf = pd.DataFrame(drift).sort_values("z_score", ascending=False).head(15)
            md += "\n\n### Drift Snapshot (Top 15 by z-score)\n"
            md += ddf.to_markdown(index=False)
        
        # Save Report
        with open(report_path, "w") as f:
            f.write(md)
        
        logger.info(f"Report generated at {report_path}")
        
        # Send Notification if alert
        if alerts and cfg.get("notifications.enable_telegram", False):
            self.send_telegram_alert(alerts)

    def send_telegram_alert(self, alerts):
        """Send alert via Telegram (placeholder)."""
        logger.info(f"Sending Telegram alert: {alerts}")
        # Implementation depends on `requests` and config tokens
