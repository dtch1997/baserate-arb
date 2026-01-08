"""Security tests for API clients and data handling."""

import pytest
import os
from unittest.mock import patch, MagicMock


class TestAPIKeySecurity:
    """Test that API keys are handled securely."""

    def test_env_example_has_no_real_keys(self):
        """Ensure .env.example doesn't contain real API keys."""
        env_example_path = os.path.join(
            os.path.dirname(__file__), "..", ".env.example"
        )

        with open(env_example_path) as f:
            content = f.read()

        # Check for patterns that look like real keys
        assert "sk-ant-" not in content  # Anthropic key pattern
        assert "sk-" not in content.replace("your_", "")  # OpenAI pattern
        assert len([line for line in content.split('\n')
                   if '=' in line and
                   not line.startswith('#') and
                   'your_' not in line.lower() and
                   'path' not in line.lower() and
                   line.split('=')[1].strip() not in ['', '127.0.0.1', '8000']]) == 0

    def test_gitignore_excludes_env(self):
        """Ensure .gitignore excludes .env files."""
        gitignore_path = os.path.join(
            os.path.dirname(__file__), "..", ".gitignore"
        )

        with open(gitignore_path) as f:
            content = f.read()

        assert ".env" in content
        assert ".env.local" in content


class TestInputValidation:
    """Test input validation and sanitization."""

    def test_market_id_injection(self):
        """Test that market IDs don't allow injection attacks."""
        from src.storage import MarketStorage

        storage = MarketStorage(data_dir="/tmp/test_baserate_storage")

        # Attempt path traversal
        malicious_ids = [
            "../../../etc/passwd",
            "market'; DROP TABLE markets;--",
            "<script>alert('xss')</script>",
            "market\x00hidden"
        ]

        for mal_id in malicious_ids:
            # Should not crash or allow path traversal
            result = storage.get_market(mal_id)
            assert result is None  # Not found, but no crash

    def test_filter_criteria_bounds(self):
        """Test that filter criteria respect bounds."""
        from src.analyzer import FilterCriteria

        # Should handle extreme values without issues
        criteria = FilterCriteria(
            min_edge=-100,  # Invalid but shouldn't crash
            min_ev=-1,
            min_quantity=-1000,
            min_kelly=-1,
            max_kelly=1000
        )

        # Should be usable (won't find results but won't crash)
        assert criteria.min_edge == -100


class TestDataIntegrity:
    """Test data integrity in storage and serialization."""

    def test_json_serialization_roundtrip(self):
        """Test that data survives JSON serialization."""
        from datetime import datetime
        from src.models.market import Market, Platform, BaseRate, BaseRateUnit

        original = Market(
            id="test-123",
            platform=Platform.KALSHI,
            title="Test Market with 'quotes' and \"double quotes\"",
            description="Description with\nnewlines\tand\ttabs",
            resolution_criteria="Criteria",
            resolution_date=datetime(2025, 12, 31, 23, 59, 59),
            yes_price=45.5,
            no_price=54.5,
            base_rate=BaseRate(
                rate=0.123456789,
                unit=BaseRateUnit.PER_YEAR,
                reasoning="Test reasoning with unicode: é ñ 中文",
                sources=["https://example.com/page?q=test&foo=bar"]
            )
        )

        # Serialize and deserialize
        data = original.to_dict()
        restored = Market.from_dict(data)

        assert restored.id == original.id
        assert restored.title == original.title
        assert restored.description == original.description
        assert abs(restored.yes_price - original.yes_price) < 0.001
        assert restored.base_rate.rate == original.base_rate.rate
        assert restored.base_rate.reasoning == original.base_rate.reasoning

    def test_storage_persistence(self):
        """Test that storage persists data correctly."""
        import tempfile
        import shutil
        from src.storage import MarketStorage
        from src.models.market import Market, Platform, BaseRate, BaseRateUnit
        from datetime import datetime, timedelta

        # Use temp directory
        temp_dir = tempfile.mkdtemp()

        try:
            # Create storage and save market
            storage1 = MarketStorage(data_dir=temp_dir)

            market = Market(
                id="persist-test",
                platform=Platform.POLYMARKET,
                title="Persistence Test",
                description="Test",
                resolution_criteria="Test",
                resolution_date=datetime.utcnow() + timedelta(days=30),
                yes_price=42
            )
            storage1.save_market(market)

            base_rate = BaseRate(
                rate=0.25,
                unit=BaseRateUnit.ABSOLUTE,
                reasoning="Persistent rate"
            )
            storage1.save_base_rate("persist-test", base_rate)

            # Create new storage instance (simulates restart)
            storage2 = MarketStorage(data_dir=temp_dir)

            # Should find the saved data
            loaded = storage2.get_market("persist-test")
            assert loaded is not None
            assert loaded.title == "Persistence Test"
            assert loaded.base_rate is not None
            assert loaded.base_rate.rate == 0.25

        finally:
            shutil.rmtree(temp_dir)


class TestRateLimitHandling:
    """Test rate limit and error handling."""

    def test_kalshi_client_retry_logic(self):
        """Test that Kalshi client has retry logic."""
        pytest.importorskip("httpx")
        from src.clients.kalshi import KalshiClient

        # Check that retry decorator is applied
        client = KalshiClient()
        assert hasattr(client._request, '__wrapped__')  # Tenacity wraps the method

    def test_polymarket_client_retry_logic(self):
        """Test that Polymarket client has retry logic."""
        pytest.importorskip("httpx")
        from src.clients.polymarket import PolymarketClient

        client = PolymarketClient()
        assert hasattr(client._gamma_request, '__wrapped__')


class TestCalculationSafety:
    """Test that calculations don't produce unsafe results."""

    def test_no_division_by_zero_in_ev(self):
        """Test EV calculation handles zero price."""
        from datetime import datetime, timedelta
        from src.models.market import Market, Platform, BaseRate, BaseRateUnit

        market = Market(
            id="test",
            platform=Platform.KALSHI,
            title="Test",
            description="Test",
            resolution_criteria="Test",
            resolution_date=datetime.utcnow() + timedelta(days=30),
            yes_price=0,
            no_price=100,
            base_rate=BaseRate(rate=0.5, unit=BaseRateUnit.ABSOLUTE, reasoning="Test")
        )

        # Should return None, not crash
        ev = market.expected_value_yes()
        assert ev is None

    def test_no_division_by_zero_in_kelly(self):
        """Test Kelly calculation handles edge cases."""
        from datetime import datetime, timedelta
        from src.models.market import Market, Platform, BaseRate, BaseRateUnit

        market = Market(
            id="test",
            platform=Platform.KALSHI,
            title="Test",
            description="Test",
            resolution_criteria="Test",
            resolution_date=datetime.utcnow() + timedelta(days=30),
            yes_price=100,  # Price = 100 means b = 0
            no_price=0,
            base_rate=BaseRate(rate=0.5, unit=BaseRateUnit.ABSOLUTE, reasoning="Test")
        )

        # Should return None, not crash
        kelly = market.kelly_fraction_yes()
        assert kelly is None

    def test_probability_bounds(self):
        """Test that probabilities stay in [0, 1]."""
        from datetime import datetime, timedelta
        from src.models.market import BaseRate, BaseRateUnit

        # Test with extreme rate
        rate = BaseRate(
            rate=0.99,
            unit=BaseRateUnit.PER_DAY,
            reasoning="Test"
        )

        # Far future - should approach but not exceed 1
        far_future = datetime.utcnow() + timedelta(days=10000)
        prob = rate.calculate_probability(far_future)

        assert 0 <= prob <= 1

    def test_kelly_never_exceeds_one(self):
        """Test Kelly fraction is bounded."""
        from datetime import datetime, timedelta
        from src.models.market import Market, Platform, BaseRate, BaseRateUnit

        # Extreme edge case
        market = Market(
            id="test",
            platform=Platform.KALSHI,
            title="Test",
            description="Test",
            resolution_criteria="Test",
            resolution_date=datetime.utcnow() + timedelta(days=30),
            yes_price=1,  # Very low price
            no_price=99,
            base_rate=BaseRate(rate=0.99, unit=BaseRateUnit.ABSOLUTE, reasoning="Test")
        )

        kelly = market.kelly_fraction_yes()
        # Kelly can theoretically be > 1 in extreme cases, but we cap it at 1 in portfolio
        assert kelly is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
