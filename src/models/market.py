"""Market and base rate data models."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import json


class Platform(Enum):
    KALSHI = "kalshi"
    POLYMARKET = "polymarket"


class BaseRateUnit(Enum):
    """Unit for base rate measurement."""
    PER_YEAR = "per_year"
    PER_MONTH = "per_month"
    PER_WEEK = "per_week"
    PER_DAY = "per_day"
    PER_EVENT = "per_event"  # e.g., per press conference, per game
    ABSOLUTE = "absolute"  # One-time probability (not time-dependent)


@dataclass
class OrderBookLevel:
    """Single level in order book."""
    price: float  # In cents (1-99)
    quantity: int  # Number of contracts


@dataclass
class MarketOrderBook:
    """Order book for a market side."""
    yes_bids: list[OrderBookLevel] = field(default_factory=list)
    yes_asks: list[OrderBookLevel] = field(default_factory=list)
    no_bids: list[OrderBookLevel] = field(default_factory=list)
    no_asks: list[OrderBookLevel] = field(default_factory=list)

    def best_yes_ask(self, min_quantity: int = 1) -> Optional[OrderBookLevel]:
        """Get best YES ask with at least min_quantity available."""
        cumulative = 0
        for level in sorted(self.yes_asks, key=lambda x: x.price):
            cumulative += level.quantity
            if cumulative >= min_quantity:
                return level
        return None

    def best_no_ask(self, min_quantity: int = 1) -> Optional[OrderBookLevel]:
        """Get best NO ask with at least min_quantity available."""
        cumulative = 0
        for level in sorted(self.no_asks, key=lambda x: x.price):
            cumulative += level.quantity
            if cumulative >= min_quantity:
                return level
        return None

    def fill_price_yes(self, quantity: int) -> Optional[float]:
        """Get average fill price to buy `quantity` YES contracts."""
        remaining = quantity
        total_cost = 0
        for level in sorted(self.yes_asks, key=lambda x: x.price):
            take = min(remaining, level.quantity)
            total_cost += take * level.price
            remaining -= take
            if remaining <= 0:
                return total_cost / quantity
        return None  # Not enough liquidity

    def fill_price_no(self, quantity: int) -> Optional[float]:
        """Get average fill price to buy `quantity` NO contracts."""
        remaining = quantity
        total_cost = 0
        for level in sorted(self.no_asks, key=lambda x: x.price):
            take = min(remaining, level.quantity)
            total_cost += take * level.price
            remaining -= take
            if remaining <= 0:
                return total_cost / quantity
        return None


@dataclass
class BaseRate:
    """Stored base rate for a market."""
    rate: float  # The base probability (0-1)
    unit: BaseRateUnit
    reasoning: str  # LLM's reasoning for this rate
    sources: list[str] = field(default_factory=list)  # URLs or references
    events_per_period: Optional[int] = None  # For per-event rates
    confidence: float = 0.5  # 0-1, how confident is the agent in this rate
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def calculate_probability(self, resolution_date: datetime) -> float:
        """
        Calculate probability adjusted for time remaining.

        For time-based rates, uses: P = 1 - (1 - r)^t
        where r is the per-period rate and t is periods remaining.
        """
        if self.unit == BaseRateUnit.ABSOLUTE:
            return self.rate

        now = datetime.utcnow()
        if resolution_date <= now:
            return self.rate  # Already resolved or resolving

        time_remaining = resolution_date - now
        days_remaining = time_remaining.total_seconds() / 86400

        # Convert to appropriate period
        if self.unit == BaseRateUnit.PER_YEAR:
            periods = days_remaining / 365.25
        elif self.unit == BaseRateUnit.PER_MONTH:
            periods = days_remaining / 30.44
        elif self.unit == BaseRateUnit.PER_WEEK:
            periods = days_remaining / 7
        elif self.unit == BaseRateUnit.PER_DAY:
            periods = days_remaining
        elif self.unit == BaseRateUnit.PER_EVENT:
            # Need to estimate events in remaining time
            if self.events_per_period:
                periods = self.events_per_period * (days_remaining / 365.25)
            else:
                periods = 1  # Default to single event if unknown
        else:
            return self.rate

        # P(at least one occurrence) = 1 - (1 - rate)^periods
        if periods <= 0:
            return 0
        return 1 - (1 - self.rate) ** periods

    def to_dict(self) -> dict:
        return {
            "rate": self.rate,
            "unit": self.unit.value,
            "reasoning": self.reasoning,
            "sources": self.sources,
            "events_per_period": self.events_per_period,
            "confidence": self.confidence,
            "last_updated": self.last_updated.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BaseRate":
        return cls(
            rate=data["rate"],
            unit=BaseRateUnit(data["unit"]),
            reasoning=data["reasoning"],
            sources=data.get("sources", []),
            events_per_period=data.get("events_per_period"),
            confidence=data.get("confidence", 0.5),
            last_updated=datetime.fromisoformat(data["last_updated"])
        )


@dataclass
class Market:
    """Unified market representation across platforms."""
    id: str
    platform: Platform
    title: str
    description: str
    resolution_criteria: str
    resolution_date: datetime
    category: str = ""

    # Current prices (in cents, 1-99)
    yes_price: float = 50
    no_price: float = 50

    # Order book (if available)
    order_book: Optional[MarketOrderBook] = None

    # Base rate (if calculated)
    base_rate: Optional[BaseRate] = None

    # Metadata
    volume: float = 0
    liquidity: float = 0
    url: str = ""
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def fair_probability(self) -> Optional[float]:
        """Get time-adjusted fair probability from base rate."""
        if not self.base_rate:
            return None
        return self.base_rate.calculate_probability(self.resolution_date)

    def market_probability(self) -> float:
        """Get implied probability from market price."""
        return self.yes_price / 100

    def edge_yes(self) -> Optional[float]:
        """Calculate edge for YES position (fair - market)."""
        fair = self.fair_probability()
        if fair is None:
            return None
        return fair - self.market_probability()

    def edge_no(self) -> Optional[float]:
        """Calculate edge for NO position (market - fair)."""
        edge = self.edge_yes()
        return -edge if edge is not None else None

    def expected_value_yes(self, buy_price: Optional[float] = None) -> Optional[float]:
        """
        Calculate expected value multiplier for YES.
        EV = (fair_prob * 100) / buy_price
        >1 means positive EV
        """
        fair = self.fair_probability()
        if fair is None:
            return None
        price = buy_price or self.yes_price
        if price <= 0:
            return None
        return (fair * 100) / price

    def expected_value_no(self, buy_price: Optional[float] = None) -> Optional[float]:
        """Calculate expected value multiplier for NO."""
        fair = self.fair_probability()
        if fair is None:
            return None
        price = buy_price or self.no_price
        if price <= 0:
            return None
        fair_no = 1 - fair
        return (fair_no * 100) / price

    def kelly_fraction_yes(self, buy_price: Optional[float] = None) -> Optional[float]:
        """
        Calculate Kelly criterion fraction for YES bet.
        f* = (bp - q) / b
        where b = odds, p = win prob, q = lose prob
        """
        fair = self.fair_probability()
        if fair is None:
            return None
        price = buy_price or self.yes_price
        if price <= 0 or price >= 100:
            return None

        # Odds: if we pay `price` cents, we get 100 cents back on win
        # So for every 1 unit risked, we get (100/price - 1) profit
        b = (100 / price) - 1  # Decimal odds minus 1
        p = fair
        q = 1 - p

        kelly = (b * p - q) / b
        return max(0, kelly)  # Don't bet negative

    def kelly_fraction_no(self, buy_price: Optional[float] = None) -> Optional[float]:
        """Calculate Kelly criterion fraction for NO bet."""
        fair = self.fair_probability()
        if fair is None:
            return None
        price = buy_price or self.no_price
        if price <= 0 or price >= 100:
            return None

        b = (100 / price) - 1
        p = 1 - fair  # Probability NO wins
        q = fair

        kelly = (b * p - q) / b
        return max(0, kelly)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "platform": self.platform.value,
            "title": self.title,
            "description": self.description,
            "resolution_criteria": self.resolution_criteria,
            "resolution_date": self.resolution_date.isoformat(),
            "category": self.category,
            "yes_price": self.yes_price,
            "no_price": self.no_price,
            "volume": self.volume,
            "liquidity": self.liquidity,
            "url": self.url,
            "last_updated": self.last_updated.isoformat(),
            "base_rate": self.base_rate.to_dict() if self.base_rate else None
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Market":
        base_rate = None
        if data.get("base_rate"):
            base_rate = BaseRate.from_dict(data["base_rate"])

        return cls(
            id=data["id"],
            platform=Platform(data["platform"]),
            title=data["title"],
            description=data["description"],
            resolution_criteria=data["resolution_criteria"],
            resolution_date=datetime.fromisoformat(data["resolution_date"]),
            category=data.get("category", ""),
            yes_price=data.get("yes_price", 50),
            no_price=data.get("no_price", 50),
            volume=data.get("volume", 0),
            liquidity=data.get("liquidity", 0),
            url=data.get("url", ""),
            last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else datetime.utcnow(),
            base_rate=base_rate
        )


@dataclass
class OpportunityAnalysis:
    """Analysis result for a market opportunity."""
    market: Market
    side: str  # "YES" or "NO"
    fair_probability: float
    market_probability: float
    edge: float  # fair - market (for YES) or market - fair (for NO)
    expected_value: float  # Multiplier
    kelly_fraction: float
    recommended_price: float  # Price at which to place limit order
    available_quantity: int  # Quantity at that price

    def to_dict(self) -> dict:
        return {
            "market_id": self.market.id,
            "platform": self.market.platform.value,
            "title": self.market.title,
            "resolution_date": self.market.resolution_date.isoformat(),
            "side": self.side,
            "fair_probability": round(self.fair_probability * 100, 2),
            "market_probability": round(self.market_probability * 100, 2),
            "edge": round(self.edge * 100, 2),
            "expected_value": round(self.expected_value, 3),
            "kelly_fraction": round(self.kelly_fraction * 100, 2),
            "recommended_price": self.recommended_price,
            "available_quantity": self.available_quantity,
            "url": self.market.url
        }
