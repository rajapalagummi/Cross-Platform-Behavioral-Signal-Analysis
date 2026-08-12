"""
ContentPulse — Main Pipeline Orchestrator
Part 1: Content diversity → 7-day return rate causal analysis
Part 2: Cross-platform podcast signal comparison

Usage:
    python main.py --mode demo      # Quick demo (200 users, 90 days)
    python main.py --mode full      # Full pipeline (1000 users, 365 days)
    python main.py --mode api       # Start dashboard server only
    python main.py --mode results   # Print saved results
"""
import os
import json
import logging
import argparse
import mlflow
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_demo():
    from data.streaming_data import run_data_pipeline
    from analysis.causal.causal_engine import run_causal_analysis
    from crossplatform.collectors.platform_collectors import collect_all_platforms
    from crossplatform.comparison.platform_comparison import run_cross_platform_analysis

    logger.info("=== ContentPulse Demo ===")
    mlflow.set_tracking_uri("sqlite:///mlruns.db")

    logger.info("\n--- Part 1: Building session dataset ---")
    sessions = run_data_pipeline(n_users=200, n_days=90)
    logger.info(f"Sessions: {len(sessions)}, Users: {sessions['user_id'].nunique()}")

    session_stats = {
        "n_sessions": int(len(sessions)),
        "n_users": int(sessions["user_id"].nunique()),
        "avg_entropy": round(float(sessions["genre_entropy"].mean()), 4),
        "median_entropy": round(float(sessions["genre_entropy"].median()), 4),
        "overall_return_rate": round(float(sessions["returned_7day"].mean()), 4),
        "high_diversity_return_rate": round(float(sessions[sessions["high_diversity"] == 1]["returned_7day"].mean()), 4),
        "low_diversity_return_rate": round(float(sessions[sessions["high_diversity"] == 0]["returned_7day"].mean()), 4),
    }
    logger.info(f"Overall return rate: {session_stats['overall_return_rate']:.4f}")
    logger.info(f"High diversity return rate: {session_stats['high_diversity_return_rate']:.4f}")
    logger.info(f"Low diversity return rate: {session_stats['low_diversity_return_rate']:.4f}")

    logger.info("\n--- Part 1: Running causal analysis ---")
    causal = run_causal_analysis(sessions)

    logger.info("\n--- Part 2: Collecting cross-platform data ---")
    platform_data = collect_all_platforms("technology")

    logger.info("\n--- Part 2: Running cross-platform comparison ---")
    crossplatform = run_cross_platform_analysis(platform_data)

    results = {
        "mode": "demo",
        "session_stats": session_stats,
        "causal": {
            "did": causal.get("did", {}),
            "psm": causal.get("psm", {}),
            "bayesian": causal.get("bayesian", {}),
            "parallel_trends": causal.get("parallel_trends", {}),
        },
        "crossplatform": crossplatform,
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("\n=== DEMO RESULTS ===")
    did = causal.get("did", {})
    bay = causal.get("bayesian", {})
    cp = crossplatform

    logger.info(f"Part 1 — Causal Analysis:")
    logger.info(f"  DiD estimate: {did.get('did_estimate', 'N/A')} (p={did.get('p_value', 'N/A')}, significant={did.get('significant', 'N/A')})")
    logger.info(f"  P(treatment > control): {bay.get('prob_treatment_better', 'N/A')}")
    logger.info(f"  Median delta: {bay.get('median_delta', 'N/A')}")
    logger.info(f"Part 2 — Cross-Platform:")
    logger.info(f"  Platforms compared: {cp.get('n_platforms', 'N/A')}")
    logger.info(f"  Topic: {cp.get('topic', 'N/A')}")

    return results


def main():
    parser = argparse.ArgumentParser(description="ContentPulse Pipeline")
    parser.add_argument("--mode", choices=["demo", "full", "api", "results"],
                        default="demo")
    args = parser.parse_args()

    if args.mode == "demo":
        results = run_demo()
        print("\n=== Final Numbers ===")
        did = results["causal"]["did"]
        bay = results["causal"]["bayesian"]
        print(f"DiD Estimate: {did.get('did_estimate')} (p={did.get('p_value')}, sig={did.get('significant')})")
        print(f"P(treatment > control): {bay.get('prob_treatment_better')}")
        print(f"Median delta: {bay.get('median_delta')}")
        print(f"HDI: [{bay.get('hdi_lower')}, {bay.get('hdi_upper')}]")
        print(f"High diversity return rate: {results['session_stats']['high_diversity_return_rate']}")
        print(f"Low diversity return rate: {results['session_stats']['low_diversity_return_rate']}")

    elif args.mode == "api":
        import uvicorn
        from api.server import app
        uvicorn.run(app, host="0.0.0.0", port=8000)

    elif args.mode == "results":
        results_path = OUTPUT_DIR / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                print(json.dumps(json.load(f), indent=2))
        else:
            print("No results found. Run --mode demo first.")


if __name__ == "__main__":
    main()
