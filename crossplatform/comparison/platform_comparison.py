"""
ContentPulse — Part 2: Cross-Platform Comparison Analysis
Compares podcast content signals across Spotify, YouTube, Apple Podcasts.

Metrics compared:
- Episode count distribution
- Engagement signals (ratings, subscribers, views)
- Content length and frequency patterns
- Genre concentration (Herfindahl-Hirschman Index)
- Platform "fingerprint" across 5 normalized dimensions
"""
import logging
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_platform_data(platform_data: dict) -> dict:
    """Normalize raw platform data into comparable DataFrames."""
    dfs = {}

    if platform_data.get("apple"):
        df = pd.DataFrame(platform_data["apple"])
        df = df.rename(columns={
            "episode_count": "content_count",
            "rating_count": "engagement_proxy",
            "avg_rating": "quality_signal",
        })
        df["platform"] = "apple"
        dfs["apple"] = df

    if platform_data.get("youtube"):
        df = pd.DataFrame(platform_data["youtube"])
        df = df.rename(columns={
            "video_count": "content_count",
            "view_count": "engagement_proxy",
            "subscriber_count": "quality_signal",
        })
        df["platform"] = "youtube"
        dfs["youtube"] = df

    if platform_data.get("spotify"):
        df = pd.DataFrame(platform_data["spotify"])
        df = df.rename(columns={
            "episode_count": "content_count",
            "followers": "engagement_proxy",
            "chart_position": "quality_signal",
        })
        df["quality_signal"] = 31 - df["quality_signal"]
        df["platform"] = "spotify"
        dfs["spotify"] = df

    return dfs


def compute_platform_fingerprint(dfs: dict) -> pd.DataFrame:
    """
    Compute 5-dimension platform fingerprint normalized to 0-1.
    Dimensions: content_volume, engagement, quality, diversity, consistency
    """
    fingerprints = []
    for platform, df in dfs.items():
        content_vol = df["content_count"].median() if "content_count" in df.columns else 0
        engagement = df["engagement_proxy"].median() if "engagement_proxy" in df.columns else 0
        quality = df["quality_signal"].median() if "quality_signal" in df.columns else 0

        if "genre" in df.columns:
            genre_counts = df["genre"].value_counts()
            total = genre_counts.sum()
            hhi = sum((c / total) ** 2 for c in genre_counts.values)
            diversity = 1 - hhi
        else:
            diversity = 0.5

        if "release_frequency_days" in df.columns:
            consistency = 1 / (1 + df["release_frequency_days"].std())
        else:
            consistency = 0.5

        fingerprints.append({
            "platform": platform,
            "content_volume_raw": content_vol,
            "engagement_raw": engagement,
            "quality_raw": quality,
            "diversity": round(diversity, 4),
            "consistency": round(consistency, 4),
        })

    fp_df = pd.DataFrame(fingerprints)

    for col in ["content_volume_raw", "engagement_raw", "quality_raw"]:
        max_val = fp_df[col].max()
        if max_val > 0:
            fp_df[col.replace("_raw", "")] = fp_df[col] / max_val
        else:
            fp_df[col.replace("_raw", "")] = 0

    return fp_df[["platform", "content_volume", "engagement",
                   "quality", "diversity", "consistency"]]


def statistical_comparison(dfs: dict) -> dict:
    """KS-test and effect sizes comparing distributions across platforms."""
    results = {}
    platforms = list(dfs.keys())

    for i in range(len(platforms)):
        for j in range(i + 1, len(platforms)):
            p1, p2 = platforms[i], platforms[j]
            pair_key = f"{p1}_vs_{p2}"

            for metric in ["content_count", "engagement_proxy"]:
                s1 = dfs[p1][metric].dropna() if metric in dfs[p1].columns else pd.Series()
                s2 = dfs[p2][metric].dropna() if metric in dfs[p2].columns else pd.Series()

                if len(s1) < 3 or len(s2) < 3:
                    continue

                ks_stat, ks_pval = stats.ks_2samp(s1, s2)
                pooled_std = np.sqrt((s1.std() ** 2 + s2.std() ** 2) / 2)
                cohens_d = (s1.mean() - s2.mean()) / (pooled_std + 1e-9)

                results[f"{pair_key}_{metric}"] = {
                    "platform_1": p1,
                    "platform_2": p2,
                    "metric": metric,
                    "ks_statistic": round(float(ks_stat), 4),
                    "ks_pvalue": round(float(ks_pval), 4),
                    "cohens_d": round(float(cohens_d), 4),
                    "significant": bool(ks_pval < 0.05),
                    "p1_median": round(float(s1.median()), 2),
                    "p2_median": round(float(s2.median()), 2),
                }

    return results


def run_cross_platform_analysis(platform_data: dict) -> dict:
    """Run full cross-platform comparison pipeline."""
    dfs = normalize_platform_data(platform_data)
    if not dfs:
        logger.warning("No platform data available for comparison")
        return {}

    fingerprint = compute_platform_fingerprint(dfs)
    statistical = statistical_comparison(dfs)

    summary = {}
    for platform, df in dfs.items():
        summary[platform] = {
            "n_shows": len(df),
            "median_content_count": round(float(df["content_count"].median()), 1)
            if "content_count" in df.columns else None,
            "median_engagement": round(float(df["engagement_proxy"].median()), 1)
            if "engagement_proxy" in df.columns else None,
        }

    fingerprint.to_csv(OUTPUT_DIR / "platform_fingerprint.csv", index=False)
    logger.info(f"\nPlatform Fingerprint:\n{fingerprint.to_string()}")

    return {
        "fingerprint": fingerprint.to_dict(orient="records"),
        "statistical_tests": statistical,
        "platform_summary": summary,
        "n_platforms": len(dfs),
        "topic": platform_data.get("topic", ""),
    }
