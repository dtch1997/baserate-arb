"""Tests for analyzer logic."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from src.models.market import (
    Market, Platform, BaseRate, BaseRateUnit,
    OrderBookLevel, MarketOrderBook, OpportunityAnalysis
)
from src.analyzer import (
    MarketAnalyzer, FilterCriteria, calculate_portfolio_kelly
)
from src.storage import MarketStorage


class TestMarketAnalyzer:
    """Tests for opportunity analysis logic."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock storage
        self.storage = MagicMock(spec=MarketStorage)
        self.analyzer = MarketAnalyzer(self.storage)

    def _create_test_market(
        self,
        market_id: str = "test",
        yes_price: float = 30,
        fair_rate: float = 0.5,
        rate_unit: BaseRateUnit = BaseRateUnit.ABSOLUTE,
        days_to_resolution: int = 30
    ) -> Market:
        """Create a test market with base rate."""
        return Market(
            id=market_id,
            platform=Platform.KALSHI,
            title=f"Test Market {market_id}",
            description="Test description",
            resolution_criteria="Test criteria",
            resolution_date=datetime.utcnow() + timedelta(days=days_to_resolution),
            yes_price=yes_price,
            no_price=100 - yes_price,
            base_rate=BaseRate(
                rate=fair_rate,
                unit=rate_unit,
                reasoning="Test reasoning"
            )
        )

    def test_analyze_market_yes_opportunity(self):
        """Test detecting YES opportunity when market underpriced."""
        # Fair prob = 50%, market = 30% -> YES is underpriced
        market = self._create_test_market(yes_price=30, fair_rate=0.5)

        opportunities = self.analyzer.analyze_market(market)

        # Should find YES opportunity
        yes_opps = [o for o in opportunities if o.side == "YES"]
        assert len(yes_opps) == 1

        opp = yes_opps[0]
        assert opp.edge > 0  # Positive edge
        assert opp.expected_value > 1  # Positive EV
        assert opp.kelly_fraction > 0  # Should bet

    def test_analyze_market_no_opportunity(self):
        """Test detecting NO opportunity when market overpriced."""
        # Fair prob = 20%, market = 50% -> NO is underpriced
        market = self._create_test_market(yes_price=50, fair_rate=0.2)

        opportunities = self.analyzer.analyze_market(market)

        # Should find NO opportunity
        no_opps = [o for o in opportunities if o.side == "NO"]
        assert len(no_opps) == 1

        opp = no_opps[0]
        assert opp.edge > 0
        assert opp.expected_value > 1
        assert opp.kelly_fraction > 0

    def test_analyze_market_no_opportunity_when_fair(self):
        """Test no opportunities when market is fairly priced."""
        # Fair prob = 50%, market = 50% -> no edge
        market = self._create_test_market(yes_price=50, fair_rate=0.5)

        opportunities = self.analyzer.analyze_market(market)

        # Should find no opportunities (no positive edge)
        assert len(opportunities) == 0

    def test_analyze_market_without_base_rate(self):
        """Test that markets without base rates return no opportunities."""
        market = Market(
            id="test",
            platform=Platform.KALSHI,
            title="Test",
            description="Test",
            resolution_criteria="Test",
            resolution_date=datetime.utcnow() + timedelta(days=30),
            yes_price=30
        )
        # No base_rate set

        opportunities = self.analyzer.analyze_market(market)
        assert len(opportunities) == 0

    def test_filter_by_min_edge(self):
        """Test filtering by minimum edge."""
        # Create markets with different edges
        market1 = self._create_test_market("m1", yes_price=30, fair_rate=0.35)  # 5% edge
        market2 = self._create_test_market("m2", yes_price=30, fair_rate=0.50)  # 20% edge

        self.storage.get_markets.return_value = [market1, market2]

        # Filter for 10% min edge, disable quantity filter
        criteria = FilterCriteria(min_edge=0.10, min_quantity=0)
        opportunities = self.analyzer.find_opportunities(criteria, min_quantity=1)

        # Only market2 should pass (20% edge > 10% threshold)
        assert len(opportunities) == 1
        assert opportunities[0].market.id == "m2"

    def test_filter_by_min_ev(self):
        """Test filtering by minimum expected value."""
        # Market with high EV: fair=60%, price=30 -> EV = 2.0
        market1 = self._create_test_market("m1", yes_price=30, fair_rate=0.6)
        # Market with low EV: fair=35%, price=30 -> EV = 1.17
        market2 = self._create_test_market("m2", yes_price=30, fair_rate=0.35)

        self.storage.get_markets.return_value = [market1, market2]

        # Filter for 1.5 min EV, disable quantity filter
        criteria = FilterCriteria(min_edge=0, min_ev=1.5, min_quantity=0)
        opportunities = self.analyzer.find_opportunities(criteria, min_quantity=1)

        # Only market1 should pass (EV 2.0 > 1.5)
        assert len(opportunities) == 1
        assert opportunities[0].market.id == "m1"

    def test_filter_by_platform(self):
        """Test filtering by platform."""
        kalshi_market = self._create_test_market("k1", yes_price=30, fair_rate=0.5)
        kalshi_market.platform = Platform.KALSHI

        poly_market = self._create_test_market("p1", yes_price=30, fair_rate=0.5)
        poly_market.platform = Platform.POLYMARKET

        self.storage.get_markets.return_value = [kalshi_market, poly_market]

        # Filter for Kalshi only, disable quantity filter
        criteria = FilterCriteria(min_edge=0, platforms=[Platform.KALSHI], min_quantity=0)
        opportunities = self.analyzer.find_opportunities(criteria, min_quantity=1)

        assert all(o.market.platform == Platform.KALSHI for o in opportunities)

    def test_filter_by_kelly_range(self):
        """Test filtering by Kelly fraction range."""
        # High Kelly opportunity
        market1 = self._create_test_market("m1", yes_price=20, fair_rate=0.6)
        # Low Kelly opportunity
        market2 = self._create_test_market("m2", yes_price=45, fair_rate=0.5)

        self.storage.get_markets.return_value = [market1, market2]

        # Filter for max 20% Kelly, disable quantity filter
        criteria = FilterCriteria(min_edge=0, min_kelly=0, max_kelly=0.20, min_quantity=0)
        opportunities = self.analyzer.find_opportunities(criteria, min_quantity=1)

        for opp in opportunities:
            assert opp.kelly_fraction <= 0.20

    def test_summary_stats(self):
        """Test summary statistics calculation."""
        market1 = self._create_test_market("m1", yes_price=30, fair_rate=0.5)
        market2 = self._create_test_market("m2", yes_price=40, fair_rate=0.6)

        opportunities = []
        for market in [market1, market2]:
            opportunities.extend(self.analyzer.analyze_market(market))

        stats = self.analyzer.get_summary_stats(opportunities)

        assert stats["count"] == len(opportunities)
        assert stats["avg_edge"] > 0
        assert stats["avg_ev"] > 1
        assert "by_platform" in stats
        assert "by_side" in stats


class TestKellyPortfolio:
    """Tests for Kelly portfolio calculations."""

    def test_kelly_portfolio_basic(self):
        """Test basic Kelly portfolio allocation."""
        # Create mock opportunities
        market = Market(
            id="test",
            platform=Platform.KALSHI,
            title="Test",
            description="Test",
            resolution_criteria="Test",
            resolution_date=datetime.utcnow() + timedelta(days=30),
            yes_price=30
        )

        opportunities = [
            OpportunityAnalysis(
                market=market,
                side="YES",
                fair_probability=0.5,
                market_probability=0.3,
                edge=0.2,
                expected_value=1.67,
                kelly_fraction=0.25,
                recommended_price=30,
                available_quantity=1000
            )
        ]

        positions = calculate_portfolio_kelly(
            opportunities,
            bankroll=10000,
            max_position_pct=0.1,
            kelly_fraction=0.5  # Half Kelly
        )

        assert "test" in positions
        pos = positions["test"]

        # Half Kelly of 25% = 12.5%, but capped at 10%
        assert pos["kelly_pct"] <= 10
        assert pos["total_cost"] <= 1000  # 10% of bankroll
        assert pos["contracts"] > 0

    def test_kelly_portfolio_respects_quantity(self):
        """Test that portfolio respects available quantity."""
        market = Market(
            id="test",
            platform=Platform.KALSHI,
            title="Test",
            description="Test",
            resolution_criteria="Test",
            resolution_date=datetime.utcnow() + timedelta(days=30),
            yes_price=30
        )

        opportunities = [
            OpportunityAnalysis(
                market=market,
                side="YES",
                fair_probability=0.5,
                market_probability=0.3,
                edge=0.2,
                expected_value=1.67,
                kelly_fraction=0.5,
                recommended_price=30,
                available_quantity=50  # Very limited
            )
        ]

        positions = calculate_portfolio_kelly(
            opportunities,
            bankroll=100000,  # Large bankroll
            max_position_pct=0.5,
            kelly_fraction=1.0
        )

        # Should be capped at available quantity
        assert positions["test"]["contracts"] <= 50


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_price_handling(self):
        """Test handling of zero prices."""
        market = Market(
            id="test",
            platform=Platform.KALSHI,
            title="Test",
            description="Test",
            resolution_criteria="Test",
            resolution_date=datetime.utcnow() + timedelta(days=30),
            yes_price=0,  # Edge case
            no_price=100,
            base_rate=BaseRate(rate=0.5, unit=BaseRateUnit.ABSOLUTE, reasoning="Test")
        )

        # Should not crash, should return None for EV
        ev = market.expected_value_yes()
        assert ev is None

    def test_expired_market(self):
        """Test handling of expired markets."""
        market = Market(
            id="test",
            platform=Platform.KALSHI,
            title="Test",
            description="Test",
            resolution_criteria="Test",
            resolution_date=datetime.utcnow() - timedelta(days=1),  # Past
            yes_price=30,
            base_rate=BaseRate(rate=0.1, unit=BaseRateUnit.PER_YEAR, reasoning="Test")
        )

        # Should still calculate something (base rate itself)
        fair = market.fair_probability()
        assert fair is not None

    def test_extreme_probabilities(self):
        """Test handling of extreme probabilities."""
        # Very low fair probability
        market = Market(
            id="test",
            platform=Platform.KALSHI,
            title="Test",
            description="Test",
            resolution_criteria="Test",
            resolution_date=datetime.utcnow() + timedelta(days=30),
            yes_price=50,
            base_rate=BaseRate(rate=0.001, unit=BaseRateUnit.ABSOLUTE, reasoning="Test")
        )

        edge = market.edge_yes()
        assert edge is not None
        assert edge < 0  # Market overpriced

    def test_kelly_negative_edge(self):
        """Test Kelly returns 0 for negative edge."""
        market = Market(
            id="test",
            platform=Platform.KALSHI,
            title="Test",
            description="Test",
            resolution_criteria="Test",
            resolution_date=datetime.utcnow() + timedelta(days=30),
            yes_price=80,
            base_rate=BaseRate(rate=0.5, unit=BaseRateUnit.ABSOLUTE, reasoning="Test")
        )

        kelly = market.kelly_fraction_yes()
        assert kelly == 0  # Don't bet on negative edge


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
