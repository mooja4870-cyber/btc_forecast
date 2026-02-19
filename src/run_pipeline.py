"""
BTC Price Prediction — MLOps Pipeline Entrypoint
================================================
Orchestrates the entire workflow:
1. Data Collection
2. Feature Engineering
3. Model Training (configured walk-forward phases)
4. Backtesting
5. Model Promotion
"""

import os
import sys
import argparse
import datetime as dt
import json
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pipeline")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import cfg
from src.data_collector import collect_all_data
from src.feature_engineer import engineer_features
from src.train_transformer import train_all_horizons
from src.actuals_collector import update_actuals
from src.monitoring import ReportGenerator

def get_run_id():
    """Generate a unique run ID based on timestamp."""
    return dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _update_latest_training_pointer(run_id: str):
    """
    Update lightweight production pointer for Transformer-only runs.
    - Always writes MODELS_DIR/LATEST.txt
    - If configured, updates MODELS_DIR/latest symlink to a run marker directory
    """
    if not run_id:
        return

    os.makedirs(cfg.models_dir, exist_ok=True)
    latest_txt = os.path.join(cfg.models_dir, "LATEST.txt")
    try:
        with open(latest_txt, "w", encoding="utf-8") as f:
            f.write(run_id)
    except Exception as e:
        logger.warning(f"Failed to write LATEST.txt: {e}")

    pointer_mode = str(cfg.promotion_config.get("latest_pointer_mode", "symlink")).lower()
    if pointer_mode != "symlink":
        logger.info(f"Latest pointer mode '{pointer_mode}' active. Skipping symlink update.")
        return

    marker_dir = os.path.join(cfg.models_dir, run_id)
    try:
        os.makedirs(marker_dir, exist_ok=True)
        marker_meta = {
            "run_id": run_id,
            "updated_at": dt.datetime.utcnow().isoformat(),
            "policy": "transformer-only pointer marker",
        }
        with open(os.path.join(marker_dir, "LATEST.txt"), "w", encoding="utf-8") as f:
            f.write(run_id)
        with open(os.path.join(marker_dir, "run_marker.json"), "w", encoding="utf-8") as f:
            json.dump(marker_meta, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to prepare run marker directory: {e}")
        return

    latest_link = os.path.join(cfg.models_dir, "latest")
    try:
        if os.path.islink(latest_link) or os.path.isfile(latest_link):
            os.remove(latest_link)
        elif os.path.isdir(latest_link):
            logger.warning(
                "models/latest is a real directory (not symlink). "
                "Symlink update skipped to avoid deleting existing files."
            )
            return
        os.symlink(marker_dir, latest_link)
        logger.info(f"Updated latest model pointer -> {marker_dir}")
    except Exception as e:
        logger.warning(f"Failed to update latest symlink pointer: {e}")


def _phase_num_from_name(phase_name: str) -> int:
    """Parse 'phaseN' -> N. Unknown names are sorted last."""
    if isinstance(phase_name, str) and phase_name.lower().startswith("phase"):
        tail = phase_name[5:]
        if tail.isdigit():
            return int(tail)
    return 10**9


def _ordered_phase_names(phases_cfg: dict = None) -> list:
    phases_cfg = phases_cfg or cfg.model_config.get("phases", {})
    names = [name for name in phases_cfg.keys() if str(name).lower().startswith("phase")]
    return sorted(names, key=_phase_num_from_name)


def _latest_validation_phase_name(phases_cfg: dict = None) -> str:
    phases_cfg = phases_cfg or cfg.model_config.get("phases", {})
    with_val = [name for name in _ordered_phase_names(phases_cfg) if phases_cfg.get(name, {}).get("val")]
    return with_val[-1] if with_val else None


def _default_phase_csv(phases_cfg: dict = None) -> str:
    phase_names = _ordered_phase_names(phases_cfg)
    nums = [str(_phase_num_from_name(name)) for name in phase_names if _phase_num_from_name(name) < 10**9]
    return ",".join(nums) if nums else "1"


def promote_model(run_id: str, results: dict, phase_to_check: str = None):
    """
    Compare current run's metrics with 'latest' and promote if better.
    """
    promotion_cfg = cfg.promotion_config
    phases_cfg = cfg.model_config.get("phases", {})
    # Ops policy: for scheduled full retraining, always point "latest"
    # to the newest successfully trained run.
    always_promote_latest = bool(promotion_cfg.get("always_promote_latest", True))
    metric_key = promotion_cfg.get("metric_key", "r2")
    min_improvement = promotion_cfg.get("improve_by_min", 0.001)
    allow_equal = promotion_cfg.get("allow_equal", True)
    
    # Use the latest phase that has validation split unless explicitly given.
    if phase_to_check is None:
        phase_to_check = _latest_validation_phase_name(phases_cfg)
    if phase_to_check is None:
        logger.warning("No validation phase configured. Skipping score-based promotion check.")
        return
    
    # Use average R2 across all horizons in the selected validation phase.
    current_metrics = results.get(phase_to_check, {})
    if not current_metrics:
        logger.warning(f"No results for {phase_to_check}, cannot evaluate for promotion.")
        return

    scores = []
    for h, res in current_metrics.items():
        # best_score is R2 in our trainer
        scores.append(res["best_score"])
    
    current_score = sum(scores) / len(scores) if scores else -float('inf')
    logger.info(f"Current Run Score ({metric_key} avg): {current_score:.4f}")

    # Check previous best
    latest_link = os.path.join(cfg.models_dir, "latest")
    previous_score = -float('inf')
    
    # If latest is a symlink or directory
    if os.path.exists(latest_link):
        # Try to read manifest from the selected comparison phase of latest
        manifest_path = os.path.join(latest_link, phase_to_check, "manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path) as f:
                    m = json.load(f)
                    # We need to reconstruct the score from the summary
                    # This implies we saved the metrics in the manifest properly
                    # Our current logic in model_trainer saves "val_metrics"
                    # We can iterate and average R2
                    prev_scores = []
                    for h_str, metrics in m.get("metrics_summary", {}).items():
                        prev_scores.append(metrics.get("r2", -float('inf')))
                    
                    if prev_scores:
                        previous_score = sum(prev_scores) / len(prev_scores)
            except Exception as e:
                logger.warning(f"Could not read previous metrics: {e}")
    
    logger.info(f"Previous Best Score: {previous_score:.4f}")
    
    # Decision
    is_better = current_score > (previous_score + min_improvement)
    is_equal_allowed = allow_equal and (current_score >= previous_score) and (abs(current_score - previous_score) < 1e-6)

    should_promote = always_promote_latest or is_better or is_equal_allowed or previous_score == -float('inf')

    if always_promote_latest:
        logger.info("Promotion policy: always_promote_latest=true (use newest retrained model).")

    if should_promote:
        logger.info(f"🚀 Promoting run {run_id} to 'latest'!")
        
        # update symlink
        # On Windows, symlinks require admin, so maybe use a pointer file if we want to be safe?
        # User is on Mac.
        try:
            if os.path.islink(latest_link) or os.path.exists(latest_link):
                os.remove(latest_link)
            
            # Symlink to the absolute path of the new run dir
            run_dir = os.path.join(cfg.models_dir, run_id)
            os.symlink(run_dir, latest_link)
            logger.info("Symlink updated.")
        except OSError as e:
            logger.error(f"Failed to create symlink: {e}")
            # Fallback to LATEST.txt
            with open(os.path.join(cfg.models_dir, "LATEST.txt"), "w") as f:
                f.write(run_id)
    else:
        logger.info(f"Run {run_id} did not improve enough over {previous_score:.4f}. Not promoting.")


def _phase_best_metric_summary(phase_results: dict) -> dict:
    """Convert horizon result dict to horizon->metrics summary."""
    out = {}
    for horizon, res in (phase_results or {}).items():
        best_name = res.get("best_model_name")
        if not best_name:
            continue
        best_metrics = res.get("results", {}).get(best_name, {})
        metrics = best_metrics.get("val_metrics") or best_metrics.get("train_metrics") or {}
        out[str(horizon)] = metrics
    return out


def _load_latest_phase_metrics_summary(phase_name: str) -> dict:
    latest_link = os.path.join(cfg.models_dir, "latest")
    manifest_path = os.path.join(latest_link, phase_name, "manifest.json")
    if not os.path.exists(manifest_path):
        return {}
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
        return manifest.get("metrics_summary", {})
    except Exception as e:
        logger.warning(f"Failed to read latest manifest for {phase_name}: {e}")
        return {}


def generate_champion_challenger_report(run_id: str, results: dict, phase_name: str):
    """Generate champion(challenger) comparison report for a phase."""
    if phase_name not in results:
        return

    current = _phase_best_metric_summary(results.get(phase_name, {}))
    previous = _load_latest_phase_metrics_summary(phase_name)
    if not current:
        return

    rows = []
    for h, cur in current.items():
        prev = previous.get(h, {}) if isinstance(previous, dict) else {}
        cur_r2 = cur.get("r2")
        prev_r2 = prev.get("r2")
        delta_r2 = (cur_r2 - prev_r2) if (cur_r2 is not None and prev_r2 is not None) else None
        rows.append({
            "horizon": int(h),
            "current_r2": cur_r2,
            "previous_r2": prev_r2,
            "delta_r2": round(delta_r2, 6) if delta_r2 is not None else None,
            "current_mape": cur.get("price_mape_pct"),
            "previous_mape": prev.get("price_mape_pct"),
            "current_direction_accuracy": cur.get("direction_accuracy"),
            "previous_direction_accuracy": prev.get("direction_accuracy"),
        })

    rows = sorted(rows, key=lambda r: r["horizon"])
    out_dir = cfg.monitoring.get("report_dir", "data/reports")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "champion_challenger_report.json")
    md_path = os.path.join(out_dir, "champion_challenger_report.md")

    with open(json_path, "w") as f:
        json.dump({
            "generated_at": dt.datetime.utcnow().isoformat(),
            "run_id": run_id,
            "phase": phase_name,
            "rows": rows,
        }, f, indent=2)

    md_lines = [
        "# Champion-Challenger Report",
        f"- Generated: {dt.datetime.utcnow().isoformat()}",
        f"- Run ID: {run_id}",
        f"- Phase: {phase_name}",
        "",
        "| Horizon | Current R2 | Previous R2 | Delta R2 | Current MAPE | Previous MAPE | Current DirAcc | Previous DirAcc |",
        "| :-- | --: | --: | --: | --: | --: | --: | --: |",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['horizon']}d | {r['current_r2']} | {r['previous_r2']} | {r['delta_r2']} | "
            f"{r['current_mape']} | {r['previous_mape']} | "
            f"{r['current_direction_accuracy']} | {r['previous_direction_accuracy']} |"
        )
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    logger.info(f"Champion-Challenger report saved: {md_path}")


def main():
    phases_cfg = cfg.model_config.get("phases", {})
    default_phases_csv = _default_phase_csv(phases_cfg)

    parser = argparse.ArgumentParser(description="BTC MLOps Pipeline")
    parser.add_argument("--skip-data", action="store_true", help="Skip data collection")
    parser.add_argument("--skip-train", action="store_true", help="Skip training")
    parser.add_argument(
        "--phases",
        type=str,
        default=default_phases_csv,
        help=f"Comma-separated phases to train (default: {default_phases_csv})",
    )
    parser.add_argument("--monitor-only", action="store_true", help="Run only monitoring and reporting")
    args = parser.parse_args()

    run_id = get_run_id()
    logger.info(f"Starting Pipeline Run: {run_id}")
    
    # ── 0. Monitoring & Reporting (Check previous performance) ──
    if cfg.monitoring.get("enabled", True):
        logger.info("Step 0: Monitoring & Reporting")
        try:
            # 1. Update Actuals
            eval_df = update_actuals()
            
            # 2. Generate Report
            if eval_df is not None and not eval_df.empty:
                gen = ReportGenerator(cfg.processed_dir, cfg.monitoring["report_dir"])
                gen.generate_report(eval_df, run_id=run_id)
        except Exception as e:
            logger.error(f"Monitoring failed: {e}")

    if args.monitor_only:
        logger.info("Monitor-only mode. Exiting.")
        sys.exit(0)
    
    # ── 1. Data Collection ──
    if not args.skip_data:
        logger.info("Step 1: Data Collection")
        collect_all_data()

    # ── 2. Feature Engineering ──
    if not args.skip_data:
        logger.info("Step 2: Feature Engineering")
        # Ensure we read the file we just created
        input_path = os.path.join(cfg.processed_dir, "merged_dataset.csv")
        if os.path.exists(input_path):
            df_raw = pd.read_csv(input_path, index_col=0, parse_dates=True)
            featured_path = os.path.join(cfg.processed_dir, "featured_dataset.csv")
            fdf = engineer_features(df_raw, save_path=featured_path)

            # Save feature-expansion status for dashboard/monitoring.
            feature_status = {
                "generated_at": dt.datetime.utcnow().isoformat(),
                "rows": int(fdf.shape[0]),
                "columns": int(fdf.shape[1]),
                "flags": {
                    "futures_term_structure": bool(cfg.feature_enabled("futures_term_structure") or cfg.futures_feature_config.get("enabled", False)),
                    "rates_expectation": bool(cfg.feature_enabled("rates_expectation") or cfg.rates_expectation_config.get("enabled", False)),
                    "geopolitical_risk": bool(cfg.feature_enabled("geopolitical_risk") or cfg.geopolitical_feature_config.get("enabled", False)),
                },
                "futures_feature_count": int(len([c for c in fdf.columns if "_fut_" in c])),
                "rates_feature_count": int(len([c for c in fdf.columns if c.startswith("rate_") or "policy_rate" in c or "fomc" in c or c.startswith("curve_")])),
                "geo_feature_count": int(len([c for c in fdf.columns if c.startswith("geo_") or c.endswith("_intensity_score") or c.endswith("_shock_score")])),
            }
            status_path = os.path.join(cfg.monitoring.get("report_dir", "data/reports"), "feature_expansion_status.json")
            os.makedirs(os.path.dirname(status_path), exist_ok=True)
            with open(status_path, "w") as f:
                json.dump(feature_status, f, indent=2)
        else:
            logger.error("Merged dataset not found. Cannot proceed.")
            sys.exit(1)

    # ── 3. Model Training ──
    results = {}
    if not args.skip_train:
        logger.info("Step 3: Model Training (Transformer-only)")
        horizons_cfg = cfg.features_config.get("horizons", [1, 2, 3, 5, 7, 15, 30, 60, 90, 180, 365])
        horizons = sorted(set(int(h) for h in horizons_cfg if int(h) > 0))
        epochs = int(cfg.model_config.get("transformer", {}).get("epochs", 15))

        train_all_horizons(horizons=horizons, epochs=epochs)
        _update_latest_training_pointer(run_id)
        logger.info("Transformer-only policy active: legacy phase/backtest/promotion steps are skipped.")

    logger.info(f"Pipeline finished successfully. Run ID: {run_id}")

if __name__ == "__main__":
    main()
