"""Polymarket API client for fetching markets and order books."""

from datetime import datetime
from typing import Optional
import json

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models.market import (
    Market, Platform, MarketOrderBook, OrderBookLevel
)


class PolymarketClient:
    """Client for Polymarket CLOB API."""

    # Polymarket endpoints
    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API = "https://clob.polymarket.com"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        api_passphrase: Optional[str] = None
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self._client = httpx.Client(timeout=30.0)

    def _get_headers(self) -> dict:
        """Get request headers."""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["POLY-API-KEY"] = self.api_key
        return headers

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _gamma_request(
        self,
        endpoint: str,
        params: Optional[dict] = None
    ) -> dict | list:
        """Make request to Gamma API (market data)."""
        url = f"{self.GAMMA_API}{endpoint}"
        response = self._client.get(url, params=params, headers=self._get_headers())
        response.raise_for_status()
        return response.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _clob_request(
        self,
        endpoint: str,
        params: Optional[dict] = None
    ) -> dict | list:
        """Make request to CLOB API (order book)."""
        url = f"{self.CLOB_API}{endpoint}"
        response = self._client.get(url, params=params, headers=self._get_headers())
        response.raise_for_status()
        return response.json()

    def get_markets(
        self,
        active: bool = True,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0,
        tag_id: Optional[int] = None
    ) -> list[dict]:
        """Get markets from Gamma API.

        Args:
            tag_id: Filter by tag ID (use get_tags() to find IDs)
        """
        params = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower(),
            "closed": str(closed).lower()
        }
        if tag_id:
            params["tag_id"] = tag_id
        return self._gamma_request("/markets", params=params)

    def get_tags(self, limit: int = 500) -> list[dict]:
        """Get available tags for filtering markets."""
        return self._gamma_request("/tags", params={"limit": limit})

    def search_tags(self, query: str) -> list[dict]:
        """Search tags by keyword."""
        tags = self.get_tags(limit=1000)
        query_lower = query.lower()
        return [t for t in tags if query_lower in t.get("label", "").lower()]

    def get_market(self, condition_id: str) -> dict:
        """Get a single market by condition ID."""
        return self._gamma_request(f"/markets/{condition_id}")

    def get_events(
        self,
        active: bool = True,
        limit: int = 100,
        offset: int = 0
    ) -> list[dict]:
        """Get events (groups of related markets)."""
        params = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower()
        }
        return self._gamma_request("/events", params=params)

    def get_event(self, event_slug: str) -> dict:
        """Get event by slug."""
        events = self._gamma_request("/events", {"slug": event_slug})
        if events:
            return events[0] if isinstance(events, list) else events
        return {}

    def parse_event_url(self, url: str) -> Optional[str]:
        """Extract event slug from Polymarket URL.

        Handles URLs like:
        - https://polymarket.com/event/will-china-invade-taiwan-before-2027
        - polymarket.com/event/will-china-invade-taiwan-before-2027
        """
        import re
        match = re.search(r'polymarket\.com/event/([^/?#]+)', url)
        return match.group(1) if match else None

    def fetch_event_markets(self, url_or_slug: str) -> list[Market]:
        """Fetch all markets from a Polymarket event URL or slug.

        Args:
            url_or_slug: Either a full URL or just the event slug

        Returns:
            List of Market objects from that event
        """
        # Extract slug if full URL provided
        if 'polymarket.com' in url_or_slug:
            slug = self.parse_event_url(url_or_slug)
            if not slug:
                raise ValueError(f"Could not parse event slug from URL: {url_or_slug}")
        else:
            slug = url_or_slug

        event = self.get_event(slug)
        if not event:
            raise ValueError(f"Event not found: {slug}")

        markets = []
        # Events contain markets in the "markets" field
        raw_markets = event.get("markets", [])

        for raw in raw_markets:
            market = self.parse_market(raw)
            markets.append(market)

        return markets

    def get_orderbook(self, token_id: str) -> dict:
        """Get order book for a token (YES or NO side of a market)."""
        return self._clob_request("/book", params={"token_id": token_id})

    def get_price(self, token_id: str) -> dict:
        """Get current price for a token."""
        return self._clob_request("/price", params={"token_id": token_id})

    def get_midpoint(self, token_id: str) -> dict:
        """Get midpoint price for a token."""
        return self._clob_request("/midpoint", params={"token_id": token_id})

    def search_markets(
        self,
        query: str,
        active: bool = True,
        limit: int = 50
    ) -> list[dict]:
        """Search markets by query string."""
        # Fetch markets and filter
        all_markets = self.get_markets(active=active, limit=500)
        query_lower = query.lower()

        matching = []
        for m in all_markets:
            question = m.get("question", "").lower()
            description = m.get("description", "").lower()
            if query_lower in question or query_lower in description:
                matching.append(m)
                if len(matching) >= limit:
                    break

        return matching

    def parse_market(self, raw: dict) -> Market:
        """Parse raw Polymarket market data into Market model."""
        # Parse end date
        end_date = raw.get("endDate") or raw.get("end_date_iso")
        if end_date:
            if isinstance(end_date, str):
                # Handle various date formats
                try:
                    resolution_date = datetime.fromisoformat(
                        end_date.replace("Z", "+00:00")
                    )
                except ValueError:
                    resolution_date = datetime.utcnow()
            else:
                resolution_date = datetime.utcnow()
        else:
            resolution_date = datetime.utcnow()

        # Get prices - Polymarket uses 0-1 scale, convert to cents
        # outcomePrices is typically a string like "[0.45, 0.55]"
        outcome_prices = raw.get("outcomePrices", "[0.5, 0.5]")
        if isinstance(outcome_prices, str):
            try:
                prices = json.loads(outcome_prices)
            except json.JSONDecodeError:
                prices = [0.5, 0.5]
        else:
            prices = outcome_prices or [0.5, 0.5]

        yes_price = float(prices[0]) * 100 if prices else 50
        no_price = float(prices[1]) * 100 if len(prices) > 1 else 100 - yes_price

        # Get token IDs for order book fetching
        tokens = raw.get("tokens", [])
        clob_token_ids = raw.get("clobTokenIds", [])

        condition_id = raw.get("conditionId", raw.get("id", ""))

        return Market(
            id=condition_id,
            platform=Platform.POLYMARKET,
            title=raw.get("question", ""),
            description=raw.get("description", ""),
            resolution_criteria=raw.get("resolutionSource", "") or raw.get("description", ""),
            resolution_date=resolution_date,
            category=raw.get("category", "") or raw.get("groupItemTitle", ""),
            yes_price=yes_price,
            no_price=no_price,
            volume=float(raw.get("volume", 0) or 0),
            liquidity=float(raw.get("liquidity", 0) or 0),
            url=f"https://polymarket.com/event/{raw.get('slug', condition_id)}",
            last_updated=datetime.utcnow()
        )

    def parse_orderbook(self, raw: dict, side: str = "yes") -> MarketOrderBook:
        """Parse raw orderbook data from CLOB."""
        orderbook = MarketOrderBook()

        bids = raw.get("bids", [])
        asks = raw.get("asks", [])

        # CLOB format: [{"price": "0.50", "size": "100"}, ...]
        if side == "yes":
            for bid in bids:
                price = float(bid.get("price", 0)) * 100
                size = int(float(bid.get("size", 0)))
                orderbook.yes_bids.append(OrderBookLevel(price=price, quantity=size))
            for ask in asks:
                price = float(ask.get("price", 0)) * 100
                size = int(float(ask.get("size", 0)))
                orderbook.yes_asks.append(OrderBookLevel(price=price, quantity=size))
        else:
            for bid in bids:
                price = float(bid.get("price", 0)) * 100
                size = int(float(bid.get("size", 0)))
                orderbook.no_bids.append(OrderBookLevel(price=price, quantity=size))
            for ask in asks:
                price = float(ask.get("price", 0)) * 100
                size = int(float(ask.get("size", 0)))
                orderbook.no_asks.append(OrderBookLevel(price=price, quantity=size))

        return orderbook

    def fetch_markets_with_prices(
        self,
        active: bool = True,
        min_liquidity: float = 0,
        limit: int = 50,
        tag_id: Optional[int] = None,
        keyword: Optional[str] = None
    ) -> list[Market]:
        """Fetch markets with current prices.

        Args:
            tag_id: Filter by tag ID (use get_tags()/search_tags() to find IDs)
            keyword: Filter by keyword in title (client-side filtering)
        """
        # Fetch more if we're filtering client-side
        fetch_limit = limit * 10 if keyword else limit * 2
        raw_markets = self.get_markets(active=active, limit=fetch_limit, tag_id=tag_id)

        markets = []
        keyword_lower = keyword.lower() if keyword else None

        for raw in raw_markets:
            liquidity = float(raw.get("liquidity", 0) or 0)
            if liquidity < min_liquidity:
                continue

            # Client-side keyword filtering
            if keyword_lower:
                question = raw.get("question", "").lower()
                description = raw.get("description", "").lower()
                if keyword_lower not in question and keyword_lower not in description:
                    continue

            market = self.parse_market(raw)
            markets.append(market)

            if len(markets) >= limit:
                break

        return markets

    def fetch_market_with_orderbook(self, condition_id: str) -> Optional[Market]:
        """Fetch a single market with its order book."""
        try:
            raw = self.get_market(condition_id)
            if not raw:
                return None

            market = self.parse_market(raw)

            # Try to get order book using CLOB token IDs
            clob_ids = raw.get("clobTokenIds", [])
            if clob_ids and len(clob_ids) >= 1:
                try:
                    yes_book = self.get_orderbook(clob_ids[0])
                    market.order_book = self.parse_orderbook(yes_book, "yes")

                    if len(clob_ids) >= 2:
                        no_book = self.get_orderbook(clob_ids[1])
                        no_parsed = self.parse_orderbook(no_book, "no")
                        market.order_book.no_bids = no_parsed.no_bids
                        market.order_book.no_asks = no_parsed.no_asks
                except Exception:
                    pass  # Order book fetch failed

            return market
        except Exception:
            return None

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
