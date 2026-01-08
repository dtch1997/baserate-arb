"""Integration and sanity check tests."""

import pytest
from datetime import datetime, timedelta
import json


class TestEndToEndFlow:
    """Test the complete flow from market to opportunity."""

    def test_complete_analysis_flow(self):
        """Test complete flow: market -> base rate -> analysis -> opportunity."""
        from src.models.market import Market, Platform, BaseRate, BaseRateUnit
        from src.storage import MarketStorage
        from src.analyzer import MarketAnalyzer, FilterCriteria
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()

        try:
            storage = MarketStorage(data_dir=temp_dir)
            analyzer = MarketAnalyzer(storage)

            # 1. Create and save a market (simulates fetch)
            market = Market(
                id="integration-test",
                platform=Platform.KALSHI,
                title="Will it rain tomorrow?",
                description="Resolves YES if precipitation > 0.1 inches",
                resolution_criteria="NOAA weather data",
                resolution_date=datetime.utcnow() + timedelta(days=1),
                yes_price=25,  # Market says 25%
                no_price=75,
                volume=10000,
                url="https://kalshi.com/markets/rain"
            )
            storage.save_market(market)

            # 2. Add base rate (simulates LLM research)
            base_rate = BaseRate(
                rate=0.4,  # Historical: 40% chance of rain
                unit=BaseRateUnit.ABSOLUTE,
                reasoning="Historical weather data shows 40% chance of rain in January",
                sources=["https://weather.gov/historical"]
            )
            storage.save_base_rate("integration-test", base_rate)

            # 3. Find opportunities (disable quantity filter since we have no order book)
            criteria = FilterCriteria(min_edge=0.05, min_ev=1.1, min_quantity=0)
            opportunities = analyzer.find_opportunities(criteria, min_quantity=1)

            # 4. Verify we found the opportunity
            assert len(opportunities) > 0

            yes_opp = next((o for o in opportunities if o.side == "YES"), None)
            assert yes_opp is not None

            # Fair = 40%, Market = 25%, Edge = 15%
            assert abs(yes_opp.fair_probability - 0.4) < 0.01
            assert abs(yes_opp.market_probability - 0.25) < 0.01
            assert abs(yes_opp.edge - 0.15) < 0.01

            # EV = 0.4 * 100 / 25 = 1.6
            assert abs(yes_opp.expected_value - 1.6) < 0.01

        finally:
            shutil.rmtree(temp_dir)


class TestBaseRateCalculations:
    """Sanity checks for base rate calculations."""

    def test_annual_rate_over_full_year(self):
        """Annual rate over a year should match the rate."""
        from src.models.market import BaseRate, BaseRateUnit

        rate = BaseRate(rate=0.10, unit=BaseRateUnit.PER_YEAR, reasoning="Test")
        one_year = datetime.utcnow() + timedelta(days=365)

        prob = rate.calculate_probability(one_year)
        # Should be close to 10%
        assert 0.09 < prob < 0.11

    def test_monthly_rate_compounds_correctly(self):
        """Monthly rate should compound over months."""
        from src.models.market import BaseRate, BaseRateUnit

        rate = BaseRate(rate=0.10, unit=BaseRateUnit.PER_MONTH, reasoning="Test")

        # 6 months out
        six_months = datetime.utcnow() + timedelta(days=182)
        prob = rate.calculate_probability(six_months)

        # 1 - (1-0.1)^6 ≈ 0.469
        assert 0.45 < prob < 0.50

    def test_per_event_rate_with_events_per_year(self):
        """Per-event rate with known events per year."""
        from src.models.market import BaseRate, BaseRateUnit

        # 2% chance per press conference, 50 per year
        rate = BaseRate(
            rate=0.02,
            unit=BaseRateUnit.PER_EVENT,
            reasoning="Test",
            events_per_period=50
        )

        one_year = datetime.utcnow() + timedelta(days=365)
        prob = rate.calculate_probability(one_year)

        # 1 - (1-0.02)^50 ≈ 0.636
        assert 0.60 < prob < 0.67

    def test_shorter_time_lower_probability(self):
        """Shorter time should mean lower probability."""
        from src.models.market import BaseRate, BaseRateUnit

        rate = BaseRate(rate=0.20, unit=BaseRateUnit.PER_YEAR, reasoning="Test")

        one_year = datetime.utcnow() + timedelta(days=365)
        half_year = datetime.utcnow() + timedelta(days=182)
        one_month = datetime.utcnow() + timedelta(days=30)

        prob_1y = rate.calculate_probability(one_year)
        prob_6m = rate.calculate_probability(half_year)
        prob_1m = rate.calculate_probability(one_month)

        assert prob_1y > prob_6m > prob_1m


class TestKellyLogic:
    """Sanity checks for Kelly criterion."""

    def test_kelly_increases_with_edge(self):
        """Higher edge should mean higher Kelly fraction."""
        from src.models.market import Market, Platform, BaseRate, BaseRateUnit

        def make_market(fair_rate):
            return Market(
                id="test",
                platform=Platform.KALSHI,
                title="Test",
                description="Test",
                resolution_criteria="Test",
                resolution_date=datetime.utcnow() + timedelta(days=30),
                yes_price=30,
                base_rate=BaseRate(rate=fair_rate, unit=BaseRateUnit.ABSOLUTE, reasoning="Test")
            )

        low_edge = make_market(0.35)   # 5% edge
        high_edge = make_market(0.50)  # 20% edge

        kelly_low = low_edge.kelly_fraction_yes()
        kelly_high = high_edge.kelly_fraction_yes()

        assert kelly_high > kelly_low

    def test_kelly_zero_for_negative_edge(self):
        """Kelly should be 0 (don't bet) for negative edge."""
        from src.models.market import Market, Platform, BaseRate, BaseRateUnit

        # Fair = 20%, Market = 30% -> negative edge on YES
        market = Market(
            id="test",
            platform=Platform.KALSHI,
            title="Test",
            description="Test",
            resolution_criteria="Test",
            resolution_date=datetime.utcnow() + timedelta(days=30),
            yes_price=30,
            base_rate=BaseRate(rate=0.20, unit=BaseRateUnit.ABSOLUTE, reasoning="Test")
        )

        kelly = market.kelly_fraction_yes()
        assert kelly == 0

    def test_half_kelly_safer(self):
        """Half Kelly should allocate less or equal to full Kelly."""
        from src.models.market import Market, Platform, BaseRate, BaseRateUnit, OpportunityAnalysis
        from src.analyzer import calculate_portfolio_kelly

        market = Market(
            id="test",
            platform=Platform.KALSHI,
            title="Test",
            description="Test",
            resolution_criteria="Test",
            resolution_date=datetime.utcnow() + timedelta(days=30),
            yes_price=30
        )

        opp = OpportunityAnalysis(
            market=market,
            side="YES",
            fair_probability=0.5,
            market_probability=0.3,
            edge=0.2,
            expected_value=1.67,
            kelly_fraction=0.25,
            recommended_price=30,
            available_quantity=10000
        )

        # Use max_position_pct high enough not to cap
        full_kelly = calculate_portfolio_kelly([opp], 10000, max_position_pct=0.5, kelly_fraction=1.0)
        half_kelly = calculate_portfolio_kelly([opp], 10000, max_position_pct=0.5, kelly_fraction=0.5)

        # Half kelly should allocate less (or equal due to int rounding)
        assert half_kelly["test"]["contracts"] <= full_kelly["test"]["contracts"]
        # And specifically, the dollar allocation should be lower
        assert half_kelly["test"]["total_cost"] <= full_kelly["test"]["total_cost"]


class TestAPIResponseParsing:
    """Test parsing of API responses."""

    def test_kalshi_market_parsing(self):
        """Test parsing Kalshi API response."""
        pytest.importorskip("httpx")
        from src.clients.kalshi import KalshiClient

        client = KalshiClient()

        raw = {
            "ticker": "RAIN-24JAN15",
            "title": "Rain in NYC on Jan 15?",
            "subtitle": "Will there be measurable rain?",
            "rules_primary": "Resolves YES if > 0.1 inches",
            "close_time": "2025-01-15T23:59:59Z",
            "yes_ask": 35,
            "no_ask": 67,
            "volume": 5000,
            "category": "weather"
        }

        market = client.parse_market(raw)

        assert market.id == "RAIN-24JAN15"
        assert market.title == "Rain in NYC on Jan 15?"
        assert market.yes_price == 35
        assert market.platform.value == "kalshi"

    def test_polymarket_market_parsing(self):
        """Test parsing Polymarket API response."""
        pytest.importorskip("httpx")
        from src.clients.polymarket import PolymarketClient

        client = PolymarketClient()

        raw = {
            "conditionId": "0x123abc",
            "question": "Will BTC hit $100k in 2025?",
            "description": "Bitcoin price milestone",
            "outcomePrices": "[0.45, 0.55]",
            "endDate": "2025-12-31T23:59:59Z",
            "volume": 1000000,
            "liquidity": 50000
        }

        market = client.parse_market(raw)

        assert market.id == "0x123abc"
        assert market.title == "Will BTC hit $100k in 2025?"
        assert market.yes_price == 45
        assert market.platform.value == "polymarket"


class TestWebAPIResponses:
    """Test web API response formats."""

    def test_opportunity_dict_format(self):
        """Test OpportunityAnalysis.to_dict() format."""
        from src.models.market import Market, Platform, OpportunityAnalysis
        from datetime import datetime, timedelta

        market = Market(
            id="test",
            platform=Platform.KALSHI,
            title="Test Market",
            description="Test",
            resolution_criteria="Test",
            resolution_date=datetime.utcnow() + timedelta(days=30),
            yes_price=30,
            url="https://kalshi.com/markets/test"
        )

        opp = OpportunityAnalysis(
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

        d = opp.to_dict()

        # Check all expected fields
        assert "market_id" in d
        assert "platform" in d
        assert "title" in d
        assert "side" in d
        assert "fair_probability" in d
        assert "market_probability" in d
        assert "edge" in d
        assert "expected_value" in d
        assert "kelly_fraction" in d
        assert "recommended_price" in d
        assert "available_quantity" in d
        assert "url" in d

        # Check values are percentages where expected
        assert d["fair_probability"] == 50.0  # Converted to %
        assert d["edge"] == 20.0  # Converted to %

        # Check JSON serializable
        json.dumps(d)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
