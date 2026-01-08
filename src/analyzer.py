"""Market analysis and opportunity detection."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.models.market import Market, OpportunityAnalysis, Platform
from src.storage import MarketStorage


@dataclass
class FilterCriteria:
    """Criteria for filtering opportunities."""
    min_edge: float = 0.02  # Minimum edge (e.g., 0.02 = 2%)
    min_ev: float = 1.05  # Minimum expected value multiplier
    max_fair_prob: float = 1.0  # Maximum fair probability
    min_fair_prob: float = 0.0  # Minimum fair probability
    min_quantity: int = 100  # Minimum available quantity at price
    min_kelly: float = 0.001  # Minimum Kelly fraction
    max_kelly: float = 1.0  # Maximum Kelly fraction
    min_confidence: float = 0.0  # Minimum base rate confidence
    # Significance filter: edge must be > min_edge_ratio * fair_prob
    # E.g., 0.5 means edge must be at least 50% of fair prob (2% fair, 1% edge won't pass)
    min_edge_ratio: float = 0.0
    platforms: Optional[list[Platform]] = None
    categories: Optional[list[str]] = None


class MarketAnalyzer:
    """Analyzes markets and identifies opportunities."""

    def __init__(self, storage: MarketStorage):
        self.storage = storage

    def analyze_market(
        self,
        market: Market,
        min_quantity: int = 1000
    ) -> list[OpportunityAnalysis]:
        """
        Analyze a single market for opportunities.

        Args:
            market: Market to analyze
            min_quantity: Minimum quantity for limit order price

        Returns:
            List of opportunities (YES and/or NO)
        """
        if not market.base_rate:
            return []

        opportunities = []
        fair_prob = market.fair_probability()

        if fair_prob is None:
            return []

        # Analyze YES side
        yes_analysis = self._analyze_side(market, "YES", fair_prob, min_quantity)
        if yes_analysis:
            opportunities.append(yes_analysis)

        # Analyze NO side
        no_analysis = self._analyze_side(market, "NO", fair_prob, min_quantity)
        if no_analysis:
            opportunities.append(no_analysis)

        return opportunities

    def _analyze_side(
        self,
        market: Market,
        side: str,
        fair_prob: float,
        min_quantity: int
    ) -> Optional[OpportunityAnalysis]:
        """Analyze one side of a market."""
        if side == "YES":
            market_prob = market.market_probability()
            edge = fair_prob - market_prob

            # Get limit order price from order book
            if market.order_book:
                level = market.order_book.best_yes_ask(min_quantity)
                if level:
                    price = level.price
                    quantity = level.quantity
                else:
                    price = market.yes_price
                    quantity = 0
            else:
                price = market.yes_price
                quantity = 0

            ev = market.expected_value_yes(price)
            kelly = market.kelly_fraction_yes(price)

        else:  # NO
            market_prob = 1 - market.market_probability()
            fair_no = 1 - fair_prob
            edge = fair_no - market_prob

            if market.order_book:
                level = market.order_book.best_no_ask(min_quantity)
                if level:
                    price = level.price
                    quantity = level.quantity
                else:
                    price = market.no_price
                    quantity = 0
            else:
                price = market.no_price
                quantity = 0

            ev = market.expected_value_no(price)
            kelly = market.kelly_fraction_no(price)
            fair_prob = fair_no  # Use NO probability for display

        if ev is None or kelly is None:
            return None

        # Only return if there's positive edge
        if edge <= 0 or ev <= 1.0:
            return None

        # For display: fair_prob was already set to fair_no for NO side above
        return OpportunityAnalysis(
            market=market,
            side=side,
            fair_probability=fair_prob,  # Already correct for both sides
            market_probability=market_prob,
            edge=edge,
            expected_value=ev,
            kelly_fraction=kelly,
            recommended_price=price,
            available_quantity=quantity
        )

    def find_opportunities(
        self,
        criteria: Optional[FilterCriteria] = None,
        min_quantity: int = 1000
    ) -> list[OpportunityAnalysis]:
        """
        Find all opportunities matching criteria.

        Args:
            criteria: Filter criteria
            min_quantity: Minimum quantity for limit order price

        Returns:
            List of opportunities sorted by expected value
        """
        if criteria is None:
            criteria = FilterCriteria()

        # Get markets with base rates
        markets = self.storage.get_markets(has_base_rate=True)

        # Filter by platform
        if criteria.platforms:
            markets = [m for m in markets if m.platform in criteria.platforms]

        # Filter by category
        if criteria.categories:
            markets = [
                m for m in markets
                if any(cat.lower() in m.category.lower() for cat in criteria.categories)
            ]

        opportunities = []

        for market in markets:
            market_opps = self.analyze_market(market, min_quantity)

            for opp in market_opps:
                # Apply filters
                if opp.edge < criteria.min_edge:
                    continue
                if opp.expected_value < criteria.min_ev:
                    continue
                if opp.fair_probability > criteria.max_fair_prob:
                    continue
                if opp.fair_probability < criteria.min_fair_prob:
                    continue
                if opp.available_quantity < criteria.min_quantity:
                    continue
                if opp.kelly_fraction < criteria.min_kelly:
                    continue
                if opp.kelly_fraction > criteria.max_kelly:
                    continue

                # Confidence filter
                if market.base_rate and market.base_rate.confidence < criteria.min_confidence:
                    continue

                # Significance filter: edge must be meaningful relative to probability
                # This filters out "2% fair vs 1% market" type noise
                if criteria.min_edge_ratio > 0 and opp.fair_probability > 0:
                    if opp.edge / opp.fair_probability < criteria.min_edge_ratio:
                        continue

                opportunities.append(opp)

        # Sort by expected value (descending)
        opportunities.sort(key=lambda x: x.expected_value, reverse=True)

        return opportunities

    def get_summary_stats(
        self,
        opportunities: list[OpportunityAnalysis]
    ) -> dict:
        """Get summary statistics for opportunities."""
        if not opportunities:
            return {
                "count": 0,
                "avg_edge": 0,
                "avg_ev": 0,
                "avg_kelly": 0,
                "total_quantity": 0,
                "by_platform": {},
                "by_side": {}
            }

        edges = [o.edge for o in opportunities]
        evs = [o.expected_value for o in opportunities]
        kellys = [o.kelly_fraction for o in opportunities]
        quantities = [o.available_quantity for o in opportunities]

        by_platform = {}
        for opp in opportunities:
            platform = opp.market.platform.value
            by_platform[platform] = by_platform.get(platform, 0) + 1

        by_side = {"YES": 0, "NO": 0}
        for opp in opportunities:
            by_side[opp.side] += 1

        return {
            "count": len(opportunities),
            "avg_edge": sum(edges) / len(edges),
            "max_edge": max(edges),
            "avg_ev": sum(evs) / len(evs),
            "max_ev": max(evs),
            "avg_kelly": sum(kellys) / len(kellys),
            "max_kelly": max(kellys),
            "total_quantity": sum(quantities),
            "by_platform": by_platform,
            "by_side": by_side
        }


def calculate_portfolio_kelly(
    opportunities: list[OpportunityAnalysis],
    bankroll: float,
    max_position_pct: float = 0.1,
    kelly_fraction: float = 0.5  # Half Kelly is common
) -> dict[str, dict]:
    """
    Calculate Kelly-optimal position sizes for a portfolio.

    Args:
        opportunities: List of opportunities
        bankroll: Total bankroll
        max_position_pct: Maximum position as fraction of bankroll
        kelly_fraction: Fraction of full Kelly to use (e.g., 0.5 for half Kelly)

    Returns:
        Dict mapping market_id to position info
    """
    positions = {}

    for opp in opportunities:
        # Scale Kelly by the chosen fraction
        kelly = opp.kelly_fraction * kelly_fraction

        # Cap at max position
        kelly = min(kelly, max_position_pct)

        # Calculate position size
        position_size = bankroll * kelly
        num_contracts = int(position_size / opp.recommended_price)

        # Cap at available quantity
        num_contracts = min(num_contracts, opp.available_quantity)

        if num_contracts > 0:
            positions[opp.market.id] = {
                "side": opp.side,
                "contracts": num_contracts,
                "price": opp.recommended_price,
                "total_cost": num_contracts * opp.recommended_price,
                "kelly_pct": kelly * 100,
                "expected_value": opp.expected_value,
                "edge": opp.edge
            }

    return positions
