"""Tests for src.engine.trade_ranker -- score calculation, ranking,
cluster detection, keyword overlap, and edge cases."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from src.models import Market, TradeCandidate, Position
from src.engine.trade_ranker import (
    _keyword_overlap,
    check_cluster_exposure,
    detect_market_clusters,
    select_best_trades,
)


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def _market(**overrides) -> Market:
    defaults = dict(
        market_id=str(uuid.uuid4()),
        question="Will X happen?",
        yes_price=0.60,
        no_price=0.40,
        market_type="political",
        keywords=["trump", "election"],
    )
    defaults.update(overrides)
    return Market(**defaults)


def _candidate(**overrides) -> TradeCandidate:
    """Build a TradeCandidate with reasonable defaults.

    Accepts ``market_overrides`` as a dict to customise the nested Market.
    """
    market_kw = overrides.pop("market_overrides", {})
    m = overrides.pop("market", None) or _market(**market_kw)
    defaults = dict(
        market=m,
        adjusted_probability=0.75,
        adjusted_confidence=0.80,
        calculated_edge=0.10,
        resolution_hours=2.0,
        position_size=200.0,
        side="BUY_YES",
        tier=1,
    )
    defaults.update(overrides)
    return TradeCandidate(**defaults)


# ---------------------------------------------------------------------------
# 1. Score calculation
# ---------------------------------------------------------------------------

class TestScoreCalculation:
    """score = edge * confidence * (1.0 / max(resolution_hours, 0.5))"""

    @patch("src.engine.trade_ranker.get_settings")
    def test_basic_score(self, mock_settings):
        """edge=0.10, confidence=0.80, resolution=2h
        time_value = 1/2 = 0.50
        score = 0.10 * 0.80 * 0.50 = 0.04
        """
        mock_settings.return_value.MAX_CLUSTER_EXPOSURE_PCT = 0.12
        c = _candidate(calculated_edge=0.10, adjusted_confidence=0.80,
                        resolution_hours=2.0)
        executed, _ = select_best_trades([c], remaining_cap=10,
                                         open_positions=[], bankroll=5000.0)
        assert len(executed) == 1
        assert pytest.approx(executed[0].score, abs=1e-9) == 0.04

    @patch("src.engine.trade_ranker.get_settings")
    def test_resolution_clamped_below_half_hour(self, mock_settings):
        """resolution=0.25h is below 0.5 floor -> time_value = 1/0.5 = 2.0"""
        mock_settings.return_value.MAX_CLUSTER_EXPOSURE_PCT = 0.12
        c = _candidate(calculated_edge=0.10, adjusted_confidence=0.80,
                        resolution_hours=0.25)
        executed, _ = select_best_trades([c], remaining_cap=10,
                                         open_positions=[], bankroll=5000.0)
        expected = 0.10 * 0.80 * 2.0  # 0.16
        assert pytest.approx(executed[0].score, abs=1e-9) == expected


# ---------------------------------------------------------------------------
# 2. Ranking
# ---------------------------------------------------------------------------

class TestRanking:

    @patch("src.engine.trade_ranker.get_settings")
    def test_ranked_by_score_descending(self, mock_settings):
        """Three candidates with different edges -> sorted by score desc."""
        mock_settings.return_value.MAX_CLUSTER_EXPOSURE_PCT = 0.12

        # Use distinct market_types so they don't cluster
        c_low = _candidate(calculated_edge=0.05, adjusted_confidence=0.80,
                            resolution_hours=2.0,
                            market_overrides={"market_type": "a", "keywords": ["aa"]})
        c_mid = _candidate(calculated_edge=0.10, adjusted_confidence=0.80,
                            resolution_hours=2.0,
                            market_overrides={"market_type": "b", "keywords": ["bb"]})
        c_high = _candidate(calculated_edge=0.20, adjusted_confidence=0.80,
                             resolution_hours=2.0,
                             market_overrides={"market_type": "c", "keywords": ["cc"]})

        executed, _ = select_best_trades(
            [c_low, c_mid, c_high], remaining_cap=10,
            open_positions=[], bankroll=5000.0,
        )
        scores = [c.score for c in executed]
        assert scores == sorted(scores, reverse=True)
        assert executed[0] is c_high
        assert executed[1] is c_mid
        assert executed[2] is c_low

    @patch("src.engine.trade_ranker.get_settings")
    def test_remaining_cap_limits_executed(self, mock_settings):
        """remaining_cap=2 -> top 2 executed, rest skipped."""
        mock_settings.return_value.MAX_CLUSTER_EXPOSURE_PCT = 0.12

        candidates = [
            _candidate(calculated_edge=0.20,
                        market_overrides={"market_type": f"type{i}", "keywords": [f"k{i}"]})
            for i in range(3)
        ]
        executed, skipped = select_best_trades(
            candidates, remaining_cap=2,
            open_positions=[], bankroll=5000.0,
        )
        assert len(executed) == 2
        assert len(skipped) == 1
        assert skipped[0].skip_reason == "daily_cap_reached"


# ---------------------------------------------------------------------------
# 3. Cluster detection
# ---------------------------------------------------------------------------

class TestClusterDetection:

    def test_same_category_close_resolution_high_overlap(self):
        """Same market_type, within 1h resolution gap, 60% keyword overlap
        -> assigned to the same cluster."""
        c1 = _candidate(resolution_hours=2.0,
                         market_overrides={
                             "market_type": "political",
                             "keywords": ["trump", "election", "vote"],
                         })
        c2 = _candidate(resolution_hours=2.5,
                         market_overrides={
                             "market_type": "political",
                             "keywords": ["trump", "election", "result"],
                         })
        clusters = detect_market_clusters([c1, c2])
        assert clusters[c1.market.market_id] == clusters[c2.market.market_id]

    def test_different_categories_different_clusters(self):
        """Different market_type -> always separate clusters."""
        c1 = _candidate(resolution_hours=2.0,
                         market_overrides={
                             "market_type": "political",
                             "keywords": ["trump", "election"],
                         })
        c2 = _candidate(resolution_hours=2.0,
                         market_overrides={
                             "market_type": "sports",
                             "keywords": ["trump", "election"],
                         })
        clusters = detect_market_clusters([c1, c2])
        assert clusters[c1.market.market_id] != clusters[c2.market.market_id]

    def test_same_category_large_resolution_gap(self):
        """Same market_type but >1h resolution gap -> different clusters."""
        c1 = _candidate(resolution_hours=2.0,
                         market_overrides={
                             "market_type": "political",
                             "keywords": ["trump", "election"],
                         })
        c2 = _candidate(resolution_hours=5.0,
                         market_overrides={
                             "market_type": "political",
                             "keywords": ["trump", "election"],
                         })
        clusters = detect_market_clusters([c1, c2])
        assert clusters[c1.market.market_id] != clusters[c2.market.market_id]


# ---------------------------------------------------------------------------
# 4. Keyword overlap (Jaccard)
# ---------------------------------------------------------------------------

class TestKeywordOverlap:

    def test_partial_overlap(self):
        """intersection=1 (trump), union=3 (trump, election, vote) -> 1/3"""
        result = _keyword_overlap(["trump", "election"], ["trump", "vote"])
        assert pytest.approx(result, abs=1e-4) == 1.0 / 3.0

    def test_empty_keyword_list(self):
        """Either list empty -> 0.0"""
        assert _keyword_overlap([], ["trump"]) == 0.0
        assert _keyword_overlap(["trump"], []) == 0.0
        assert _keyword_overlap([], []) == 0.0


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    @patch("src.engine.trade_ranker.get_settings")
    def test_zero_candidates(self, mock_settings):
        """No candidates -> ([], [])."""
        executed, skipped = select_best_trades(
            [], remaining_cap=10, open_positions=[], bankroll=5000.0,
        )
        assert executed == []
        assert skipped == []

    @patch("src.engine.trade_ranker.get_settings")
    def test_single_candidate(self, mock_settings):
        """One candidate -> ([candidate], [])."""
        mock_settings.return_value.MAX_CLUSTER_EXPOSURE_PCT = 0.12
        c = _candidate()
        executed, skipped = select_best_trades(
            [c], remaining_cap=10, open_positions=[], bankroll=5000.0,
        )
        assert len(executed) == 1
        assert executed[0] is c
        assert skipped == []


# ---------------------------------------------------------------------------
# 2. Faster resolution -> higher time_value
# ---------------------------------------------------------------------------

class TestTimeValue:

    @patch("src.engine.trade_ranker.get_settings")
    def test_faster_resolution_higher_time_value(self, mock_settings):
        """1h resolution vs 24h resolution: 1h should have higher score
        because time_value = 1/max(resolution, 0.5).
        1h  -> time_value = 1/1  = 1.0  -> score = 0.10 * 0.80 * 1.0  = 0.08
        24h -> time_value = 1/24 = 0.0417 -> score = 0.10 * 0.80 * 0.0417 = 0.00333
        """
        mock_settings.return_value.MAX_CLUSTER_EXPOSURE_PCT = 0.12

        c_fast = _candidate(calculated_edge=0.10, adjusted_confidence=0.80,
                            resolution_hours=1.0,
                            market_overrides={"market_type": "a", "keywords": ["aa"]})
        c_slow = _candidate(calculated_edge=0.10, adjusted_confidence=0.80,
                            resolution_hours=24.0,
                            market_overrides={"market_type": "b", "keywords": ["bb"]})

        executed, _ = select_best_trades(
            [c_slow, c_fast], remaining_cap=10,
            open_positions=[], bankroll=5000.0,
        )
        # fast should be ranked first due to higher time_value
        assert executed[0] is c_fast
        assert executed[1] is c_slow

        expected_fast = 0.10 * 0.80 * (1.0 / 1.0)   # 0.08
        expected_slow = 0.10 * 0.80 * (1.0 / 24.0)   # ~0.003333
        assert pytest.approx(c_fast.score, abs=1e-9) == expected_fast
        assert pytest.approx(c_slow.score, abs=1e-6) == expected_slow
        assert c_fast.score > c_slow.score


# ---------------------------------------------------------------------------
# 6. remaining_cap=0 -> all candidates get skip_reason
# ---------------------------------------------------------------------------

class TestRemainingCapZero:

    @patch("src.engine.trade_ranker.get_settings")
    def test_remaining_cap_zero_all_skipped(self, mock_settings):
        """remaining_cap=0 -> every candidate is skipped with daily_cap_reached."""
        mock_settings.return_value.MAX_CLUSTER_EXPOSURE_PCT = 0.12

        candidates = [
            _candidate(calculated_edge=0.20,
                       market_overrides={"market_type": f"type{i}", "keywords": [f"k{i}"]})
            for i in range(3)
        ]
        executed, skipped = select_best_trades(
            candidates, remaining_cap=0,
            open_positions=[], bankroll=5000.0,
        )
        assert len(executed) == 0
        assert len(skipped) == 3
        for s in skipped:
            assert s.skip_reason == "daily_cap_reached"


# ---------------------------------------------------------------------------
# 8. Same category, close resolution, 20% keyword overlap -> different clusters
# ---------------------------------------------------------------------------

class TestLowKeywordOverlapDifferentClusters:

    def test_same_category_close_resolution_low_overlap(self):
        """Same market_type, within 1h resolution gap, only 20% keyword overlap
        (below 50% threshold) -> assigned to different clusters."""
        # keywords: ["fed", "rates", "fomc", "inflation", "economy"]
        # vs        ["fed", "crypto", "bitcoin", "altcoin", "defi"]
        # intersection = {"fed"} = 1
        # union = {"fed","rates","fomc","inflation","economy","crypto","bitcoin","altcoin","defi"} = 9
        # Jaccard = 1/9 ~ 0.111 < 0.50
        c1 = _candidate(resolution_hours=2.0,
                         market_overrides={
                             "market_type": "economic",
                             "keywords": ["fed", "rates", "fomc", "inflation", "economy"],
                         })
        c2 = _candidate(resolution_hours=2.5,
                         market_overrides={
                             "market_type": "economic",
                             "keywords": ["fed", "crypto", "bitcoin", "altcoin", "defi"],
                         })
        clusters = detect_market_clusters([c1, c2])
        assert clusters[c1.market.market_id] != clusters[c2.market.market_id]


# ---------------------------------------------------------------------------
# 12. Specific Jaccard: ["fed","rates","fomc"] vs ["fed","rates","cut"] -> 2/4
# ---------------------------------------------------------------------------

class TestSpecificJaccard:

    def test_fed_rates_jaccard(self):
        """intersection = {"fed","rates"} = 2
        union = {"fed","rates","fomc","cut"} = 4
        Jaccard = 2/4 = 0.50
        """
        result = _keyword_overlap(["fed", "rates", "fomc"], ["fed", "rates", "cut"])
        assert pytest.approx(result, abs=1e-9) == 0.50


# ---------------------------------------------------------------------------
# 14-16. Cluster exposure checks
# ---------------------------------------------------------------------------

class TestClusterExposure:

    @patch("src.engine.trade_ranker.get_settings")
    def test_exceeds_cluster_limit_small_bankroll(self, mock_settings):
        """Cluster with $500 existing + $100 pending + $200 new, bankroll=$5000
        MAX_CLUSTER_EXPOSURE_PCT = 0.12 -> limit = 600
        total = 500 + 100 + 200 = 800 > 600 -> exceeds
        """
        mock_settings.return_value.MAX_CLUSTER_EXPOSURE_PCT = 0.12

        cluster_id = "cluster_1"
        clusters = {"m_existing": cluster_id, "m_pending": cluster_id, "m_new": cluster_id}

        existing_pos = [Position(market_id="m_existing", side="BUY_YES",
                                 entry_price=0.6, size_usd=500.0)]
        pending_candidate = _candidate(
            position_size=100.0,
            market_overrides={"market_id": "m_pending", "market_type": "economic",
                              "keywords": ["fed"]},
        )
        new_candidate = _candidate(
            position_size=200.0,
            market_overrides={"market_id": "m_new", "market_type": "economic",
                              "keywords": ["fed"]},
        )

        result = check_cluster_exposure(
            new_candidate, cluster_id, existing_pos, [pending_candidate],
            clusters, bankroll=5000.0,
        )
        assert result is False  # 800 > 600

    @patch("src.engine.trade_ranker.get_settings")
    def test_within_cluster_limit_large_bankroll(self, mock_settings):
        """Same amounts ($500 existing + $100 pending + $200 new) but bankroll=$10000
        MAX_CLUSTER_EXPOSURE_PCT = 0.12 -> limit = 1200
        total = 800 < 1200 -> within limit
        """
        mock_settings.return_value.MAX_CLUSTER_EXPOSURE_PCT = 0.12

        cluster_id = "cluster_1"
        clusters = {"m_existing": cluster_id, "m_pending": cluster_id, "m_new": cluster_id}

        existing_pos = [Position(market_id="m_existing", side="BUY_YES",
                                 entry_price=0.6, size_usd=500.0)]
        pending_candidate = _candidate(
            position_size=100.0,
            market_overrides={"market_id": "m_pending", "market_type": "economic",
                              "keywords": ["fed"]},
        )
        new_candidate = _candidate(
            position_size=200.0,
            market_overrides={"market_id": "m_new", "market_type": "economic",
                              "keywords": ["fed"]},
        )

        result = check_cluster_exposure(
            new_candidate, cluster_id, existing_pos, [pending_candidate],
            clusters, bankroll=10000.0,
        )
        assert result is True  # 800 < 1200

    @patch("src.engine.trade_ranker.get_settings")
    def test_no_existing_exposure_passes(self, mock_settings):
        """No existing exposure in cluster -> passes if position alone is small enough.
        candidate.position_size=200, bankroll=5000, limit=600 -> 200 < 600 -> passes.
        """
        mock_settings.return_value.MAX_CLUSTER_EXPOSURE_PCT = 0.12

        cluster_id = "cluster_1"
        clusters = {"m_new": cluster_id}

        new_candidate = _candidate(
            position_size=200.0,
            market_overrides={"market_id": "m_new", "market_type": "economic",
                              "keywords": ["fed"]},
        )

        result = check_cluster_exposure(
            new_candidate, cluster_id, open_positions=[], pending=[],
            clusters=clusters, bankroll=5000.0,
        )
        assert result is True  # 200 < 600


# ---------------------------------------------------------------------------
# 19. All candidates in same cluster -> only first fits, rest skipped
# ---------------------------------------------------------------------------

class TestAllSameCluster:

    @patch("src.engine.trade_ranker.get_settings")
    def test_all_same_cluster_only_first_fits(self, mock_settings):
        """All candidates share the same cluster (same category, close resolution,
        high keyword overlap). Each has position_size=500, bankroll=5000,
        MAX_CLUSTER_EXPOSURE_PCT=0.12 -> limit=600.
        First candidate: 500 <= 600 -> fits.
        Second candidate: 500 + 500 = 1000 > 600 -> skipped.
        Third candidate: similarly skipped.
        """
        mock_settings.return_value.MAX_CLUSTER_EXPOSURE_PCT = 0.12

        shared_kw = ["trump", "election", "vote"]
        c1 = _candidate(calculated_edge=0.20, position_size=500.0,
                         resolution_hours=2.0,
                         market_overrides={"market_type": "political",
                                           "keywords": shared_kw})
        c2 = _candidate(calculated_edge=0.15, position_size=500.0,
                         resolution_hours=2.0,
                         market_overrides={"market_type": "political",
                                           "keywords": shared_kw})
        c3 = _candidate(calculated_edge=0.10, position_size=500.0,
                         resolution_hours=2.0,
                         market_overrides={"market_type": "political",
                                           "keywords": shared_kw})

        executed, skipped = select_best_trades(
            [c1, c2, c3], remaining_cap=10,
            open_positions=[], bankroll=5000.0,
        )
        assert len(executed) == 1
        assert executed[0] is c1  # highest edge
        assert len(skipped) == 2
        for s in skipped:
            assert s.skip_reason == "cluster_exposure_limit"
