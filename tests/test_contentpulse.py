"""ContentPulse — Unit Tests"""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.streaming_data import generate_user_profiles, generate_listening_events, build_sessions, compute_session_features
from crossplatform.comparison.platform_comparison import normalize_platform_data, compute_platform_fingerprint


class TestDataGeneration:
    def test_user_profiles_shape(self):
        users = generate_user_profiles(10)
        assert len(users) == 10
        assert "user_id" in users.columns
        assert "diversity_tendency" in users.columns

    def test_diversity_tendency_bounded(self):
        users = generate_user_profiles(50)
        assert (users["diversity_tendency"] >= 0).all()
        assert (users["diversity_tendency"] <= 1).all()

    def test_listening_events_generated(self):
        users = generate_user_profiles(5)
        events = generate_listening_events(users, n_days=30)
        assert len(events) > 0
        assert "genre" in events.columns
        assert "duration_sec" in events.columns

    def test_sessions_built(self):
        users = generate_user_profiles(5)
        events = generate_listening_events(users, n_days=30)
        sessions = build_sessions(events)
        assert "session_id" in sessions.columns

    def test_genre_entropy_non_negative(self):
        users = generate_user_profiles(10)
        events = generate_listening_events(users, n_days=60)
        sessions = build_sessions(events)
        features = compute_session_features(sessions)
        assert (features["genre_entropy"] >= 0).all()

    def test_return_rate_binary(self):
        users = generate_user_profiles(10)
        events = generate_listening_events(users, n_days=60)
        sessions = build_sessions(events)
        features = compute_session_features(sessions)
        assert features["returned_7day"].isin([0, 1]).all()


class TestCrossPlatform:
    def setup_method(self):
        self.platform_data = {
            "apple": [{"show_name": f"Show {i}", "episode_count": 50 + i,
                       "rating_count": 100 * i, "avg_rating": 4.0,
                       "genre": "technology", "platform": "apple"} for i in range(10)],
            "youtube": [{"show_name": f"Channel {i}", "video_count": 30 + i,
                         "view_count": 10000 * i, "subscriber_count": 1000 * i,
                         "genre": "podcast", "platform": "youtube"} for i in range(10)],
            "spotify": [{"show_name": f"Podcast {i}", "episode_count": 40 + i,
                         "followers": 5000 * i, "chart_position": i + 1,
                         "genre": "technology", "platform": "spotify"} for i in range(10)],
        }

    def test_normalize_returns_dfs(self):
        dfs = normalize_platform_data(self.platform_data)
        assert len(dfs) == 3
        assert "apple" in dfs
        assert "youtube" in dfs

    def test_fingerprint_dimensions(self):
        dfs = normalize_platform_data(self.platform_data)
        fp = compute_platform_fingerprint(dfs)
        assert len(fp) == 3
        assert "content_volume" in fp.columns
        assert "engagement" in fp.columns

    def test_fingerprint_values_bounded(self):
        dfs = normalize_platform_data(self.platform_data)
        fp = compute_platform_fingerprint(dfs)
        for col in ["content_volume", "engagement", "diversity", "consistency"]:
            assert (fp[col] >= 0).all()
            assert (fp[col] <= 1).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
