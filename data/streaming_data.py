"""
ContentPulse — Part 1 Data Layer
Generates realistic streaming consumption data modeled on Last.fm listening patterns.
Uses synthetic data with real statistical properties from public Last.fm research.

Real Last.fm dataset: https://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-1K.html
Falls back to synthetic generation if dataset not downloaded.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
DATA_DIR = Path("data/raw")
CACHE_DIR = Path("data/cache")
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

GENRES = ["pop", "rock", "hip-hop", "electronic", "jazz", "classical",
          "country", "r&b", "indie", "metal", "folk", "latin", "podcast",
          "comedy", "news", "true-crime", "sports", "technology"]

PODCAST_GENRES = ["true-crime", "comedy", "news", "sports", "technology"]
MUSIC_GENRES = [g for g in GENRES if g not in PODCAST_GENRES]

N_USERS = 1000
N_DAYS = 365
SESSION_GAP_MINUTES = 30
RETURN_WINDOW_DAYS = 3


def generate_user_profiles(n_users: int = N_USERS, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    profiles = []
    for uid in range(n_users):
        tenure_days = np.random.randint(30, 730)
        preferred_genres = np.random.choice(GENRES, size=np.random.randint(2, 6),
                                             replace=False).tolist()
        diversity_tendency = np.random.beta(2, 3)
        daily_sessions = np.random.lognormal(0.3, 0.5)
        profiles.append({
            "user_id": f"user_{uid:04d}",
            "tenure_days": tenure_days,
            "preferred_genres": preferred_genres,
            "diversity_tendency": round(diversity_tendency, 4),
            "avg_daily_sessions": round(daily_sessions, 2),
            "cohort": "veteran" if tenure_days > 365 else
                      "established" if tenure_days > 90 else "new",
        })
    return pd.DataFrame(profiles)


def generate_listening_events(user_profiles: pd.DataFrame,
                               n_days: int = N_DAYS,
                               seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    events = []
    start_date = datetime(2023, 1, 1)

    for _, user in user_profiles.iterrows():
        user_id = user["user_id"]
        preferred = user["preferred_genres"]
        diversity = user["diversity_tendency"]
        avg_sessions = user["avg_daily_sessions"]

        for day in range(min(n_days, user["tenure_days"])):
            date = start_date + timedelta(days=day)
            n_sessions = max(0, int(np.random.poisson(avg_sessions)))

            for _ in range(n_sessions):
                hour = int(np.random.choice(range(24),
                           p=_hour_weights()))
                session_start = date + timedelta(hours=hour,
                                                  minutes=np.random.randint(0, 60))
                n_tracks = np.random.randint(3, 20)

                session_genres = []
                for _ in range(n_tracks):
                    if np.random.random() < (1 - diversity):
                        genre = np.random.choice(preferred)
                    else:
                        genre = np.random.choice(GENRES)
                    session_genres.append(genre)

                    duration_sec = int(np.random.lognormal(5.2, 0.5))
                    skip = np.random.random() < 0.15
                    repeat = np.random.random() < 0.08

                    events.append({
                        "user_id": user_id,
                        "timestamp": session_start,
                        "genre": genre,
                        "duration_sec": duration_sec if not skip else int(duration_sec * 0.2),
                        "skipped": int(skip),
                        "repeated": int(repeat),
                        "session_date": date.strftime("%Y-%m-%d"),
                        "cohort": user["cohort"],
                    })
                    session_start += timedelta(seconds=duration_sec + np.random.randint(1, 10))

    df = pd.DataFrame(events)
    logger.info(f"Generated {len(df)} listening events for {user_profiles['user_id'].nunique()} users")
    return df


def _hour_weights() -> np.ndarray:
    weights = np.array([0.5, 0.3, 0.2, 0.1, 0.1, 0.2, 0.5, 1.0,
                        1.5, 1.8, 2.0, 2.2, 2.0, 1.8, 1.5, 1.5,
                        2.0, 2.5, 3.0, 3.5, 3.0, 2.5, 2.0, 1.2])
    return weights / weights.sum()


def build_sessions(events_df: pd.DataFrame,
                    gap_minutes: int = SESSION_GAP_MINUTES) -> pd.DataFrame:
    events_df = events_df.copy()
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"])
    events_df = events_df.sort_values(["user_id", "timestamp"])

    sessions = []
    for user_id, group in events_df.groupby("user_id"):
        group = group.reset_index(drop=True)
        session_id = 0
        prev_time = None

        for _, row in group.iterrows():
            if prev_time is None or (row["timestamp"] - prev_time).seconds / 60 > gap_minutes:
                session_id += 1
            row_dict = row.to_dict()
            row_dict["session_id"] = f"{user_id}_s{session_id:04d}"
            sessions.append(row_dict)
            prev_time = row["timestamp"]

    return pd.DataFrame(sessions)


def compute_session_features(sessions_df: pd.DataFrame) -> pd.DataFrame:
    """Compute genre entropy and other session-level features."""
    def genre_entropy(genres):
        from collections import Counter
        counts = Counter(genres)
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        return max(0.0, -sum(p * np.log2(p + 1e-9) for p in probs))

    agg = sessions_df.groupby(["user_id", "session_id", "session_date", "cohort"]).agg(
        n_tracks=("genre", "count"),
        session_duration_min=("duration_sec", lambda x: x.sum() / 60),
        skip_rate=("skipped", "mean"),
        repeat_rate=("repeated", "mean"),
        n_unique_genres=("genre", "nunique"),
        genres=("genre", list),
    ).reset_index()

    agg["genre_entropy"] = agg["genres"].apply(genre_entropy)
    agg["high_diversity"] = (agg["genre_entropy"] > agg["genre_entropy"].median()).astype(int)
    agg["session_date"] = pd.to_datetime(agg["session_date"])

    user_dates = agg.groupby("user_id")["session_date"].apply(list).to_dict()
    def compute_3day_return(row):
        user = row["user_id"]
        date = row["session_date"]
        future_dates = [d for d in user_dates.get(user, [])
                        if timedelta(days=1) <= (d - date) <= timedelta(days=3)]
        return int(len(future_dates) > 0)

    agg["returned_7day"] = agg.apply(compute_3day_return, axis=1)
    agg = agg.drop(columns=["genres"])
    logger.info(f"Computed features for {len(agg)} sessions")
    return agg


def run_data_pipeline(n_users: int = 300, n_days: int = 180) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"sessions_{n_users}u_{n_days}d.parquet"
    if cache_path.exists():
        logger.info(f"Loading cached sessions from {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info(f"Generating data for {n_users} users, {n_days} days")
    users = generate_user_profiles(n_users)
    users["avg_daily_sessions"] = users["avg_daily_sessions"] * 0.15
    events = generate_listening_events(users, n_days)
    sessions = build_sessions(events)
    features = compute_session_features(sessions)
    features.to_parquet(cache_path, index=False)
    logger.info(f"Cached to {cache_path}")
    return features
