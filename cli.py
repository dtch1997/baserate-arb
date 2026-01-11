#!/usr/bin/env python3
"""Command-line interface for Base Rate Arbitrage Scanner."""

import argparse
import json
import os
import sys
from datetime import datetime
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.models.market import Platform
from src.storage import MarketStorage
from src.analyzer import MarketAnalyzer, FilterCriteria
from src.clients.kalshi import KalshiClient
from src.clients.polymarket import PolymarketClient
from src.agents.base_rate_agent import BaseRateAgent


def cmd_fetch(args):
    """Fetch markets from exchanges."""
    storage = MarketStorage()

    # Handle URL-based fetching for specific events
    if args.url:
        print(f"Fetching markets from URL: {args.url}")
        with PolymarketClient() as client:
            try:
                markets = client.fetch_event_markets(args.url)
                storage.save_markets(markets)
                print(f"  Saved {len(markets)} markets from event")
                for m in markets:
                    print(f"    - {m.title[:60]}...")
            except ValueError as e:
                print(f"Error: {e}")
                return
        print(f"\nTotal markets in storage: {storage.market_count}")
        return

    if args.platform in ("kalshi", "all"):
        print("Fetching Kalshi markets...")
        api_key = os.getenv("KALSHI_API_KEY")
        pk_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")

        with KalshiClient(api_key=api_key, private_key_path=pk_path) as client:
            markets = client.fetch_markets_with_books(
                min_volume=args.min_volume,
                limit=args.limit
            )
            storage.save_markets(markets)
            print(f"  Saved {len(markets)} Kalshi markets")

    if args.platform in ("polymarket", "all"):
        tag_info = f" (tag_id={args.tag_id})" if args.tag_id else ""
        keyword_info = f" (keyword={args.keyword})" if args.keyword else ""
        print(f"Fetching Polymarket markets{tag_info}{keyword_info}...")
        with PolymarketClient() as client:
            markets = client.fetch_markets_with_prices(
                min_liquidity=args.min_volume,
                limit=args.limit,
                tag_id=args.tag_id,
                keyword=args.keyword
            )
            storage.save_markets(markets)
            print(f"  Saved {len(markets)} Polymarket markets")

    print(f"\nTotal markets in storage: {storage.market_count}")


def cmd_research(args):
    """Research base rates for markets."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        return

    storage = MarketStorage()

    # Handle URL-based research (fetch + research in one step)
    if args.url:
        print(f"Fetching markets from URL: {args.url}")
        with PolymarketClient() as client:
            try:
                markets = client.fetch_event_markets(args.url)
                storage.save_markets(markets)
                print(f"  Found {len(markets)} markets")
            except ValueError as e:
                print(f"Error: {e}")
                return
    elif args.market_id:
        markets = [storage.get_market(args.market_id)]
        if not markets[0]:
            print(f"Market {args.market_id} not found")
            return
    else:
        markets = storage.get_markets(has_base_rate=False if not args.include_existing else None)
        markets = markets[:args.limit]

    print(f"Researching base rates for {len(markets)} markets...")

    with BaseRateAgent(api_key=api_key) as agent:
        for i, market in enumerate(markets, 1):
            print(f"\n[{i}/{len(markets)}] {market.title[:60]}...")
            try:
                base_rate = agent.research_base_rate(market)
                if base_rate:
                    storage.save_base_rate(market.id, base_rate)
                    print(f"  -> Rate: {base_rate.rate:.4f} ({base_rate.unit.value})")
                else:
                    print("  -> Could not determine base rate")
            except Exception as e:
                print(f"  -> Error: {e}")

    print(f"\nMarkets with base rates: {storage.base_rate_count}")


def cmd_opportunities(args):
    """Show trading opportunities."""
    storage = MarketStorage()
    analyzer = MarketAnalyzer(storage)

    criteria = FilterCriteria(
        min_edge=args.min_edge / 100,
        min_ev=args.min_ev,
        min_quantity=args.min_quantity,
        min_kelly=args.min_kelly / 100,
        platforms=[Platform(args.platform)] if args.platform else None
    )

    opportunities = analyzer.find_opportunities(criteria, min_quantity=args.min_quantity)

    if not opportunities:
        print("No opportunities found matching criteria.")
        return

    # Format for display
    rows = []
    for opp in opportunities[:args.limit]:
        rows.append([
            opp.market.title[:40] + "..." if len(opp.market.title) > 40 else opp.market.title,
            opp.market.platform.value,
            opp.side,
            f"{opp.fair_probability*100:.1f}%",
            f"{opp.market_probability*100:.1f}%",
            f"{opp.edge*100:.1f}%",
            f"{opp.expected_value:.2f}x",
            f"{opp.kelly_fraction*100:.1f}%",
            f"¢{opp.recommended_price:.0f}",
            opp.available_quantity
        ])

    headers = ["Market", "Platform", "Side", "Fair", "Market", "Edge", "EV", "Kelly", "Price", "Qty"]
    print(tabulate(rows, headers=headers, tablefmt="simple"))

    # Summary stats
    stats = analyzer.get_summary_stats(opportunities)
    print(f"\n{stats['count']} opportunities | "
          f"Avg Edge: {stats['avg_edge']*100:.1f}% | "
          f"Avg EV: {stats['avg_ev']:.2f}x | "
          f"Avg Kelly: {stats['avg_kelly']*100:.1f}%")


def cmd_export(args):
    """Export opportunities to JSON."""
    storage = MarketStorage()
    analyzer = MarketAnalyzer(storage)

    criteria = FilterCriteria(
        min_edge=args.min_edge / 100,
        min_ev=args.min_ev,
        min_quantity=args.min_quantity
    )

    opportunities = analyzer.find_opportunities(criteria)

    data = {
        "generated_at": datetime.utcnow().isoformat(),
        "criteria": {
            "min_edge": args.min_edge,
            "min_ev": args.min_ev,
            "min_quantity": args.min_quantity
        },
        "opportunities": [opp.to_dict() for opp in opportunities],
        "stats": analyzer.get_summary_stats(opportunities)
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Exported {len(opportunities)} opportunities to {args.output}")
    else:
        print(json.dumps(data, indent=2))


def cmd_serve(args):
    """Start the web server."""
    import uvicorn
    print(f"Starting server at http://{args.host}:{args.port}")
    uvicorn.run("src.web.app:app", host=args.host, port=args.port, reload=args.reload)


def cmd_tags(args):
    """List or search Polymarket tags."""
    with PolymarketClient() as client:
        if args.search:
            tags = client.search_tags(args.search)
            print(f"Tags matching '{args.search}':")
        else:
            tags = client.get_tags(limit=args.limit)
            print(f"Polymarket tags (showing {len(tags)}):")

        rows = [(t["id"], t["label"]) for t in tags[:args.limit]]
        print(tabulate(rows, headers=["ID", "Label"], tablefmt="simple"))


def main():
    parser = argparse.ArgumentParser(description="Base Rate Arbitrage Scanner")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Fetch command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch markets from exchanges")
    fetch_parser.add_argument("--platform", choices=["kalshi", "polymarket", "all"], default="all")
    fetch_parser.add_argument("--min-volume", type=float, default=0)
    fetch_parser.add_argument("--limit", type=int, default=100)
    fetch_parser.add_argument("--tag-id", type=int, help="Filter by tag ID (use 'tags' command to list)")
    fetch_parser.add_argument("--keyword", help="Filter by keyword in title (e.g., 'China')")
    fetch_parser.add_argument("--url", help="Fetch from specific Polymarket event URL")
    fetch_parser.set_defaults(func=cmd_fetch)

    # Research command
    research_parser = subparsers.add_parser("research", help="Research base rates")
    research_parser.add_argument("--market-id", help="Research specific market")
    research_parser.add_argument("--url", help="Fetch and research markets from Polymarket event URL")
    research_parser.add_argument("--limit", type=int, default=10)
    research_parser.add_argument("--include-existing", action="store_true")
    research_parser.set_defaults(func=cmd_research)

    # Opportunities command
    opp_parser = subparsers.add_parser("opportunities", help="Show trading opportunities")
    opp_parser.add_argument("--min-edge", type=float, default=2, help="Min edge in percent")
    opp_parser.add_argument("--min-ev", type=float, default=1.05, help="Min expected value")
    opp_parser.add_argument("--min-quantity", type=int, default=100)
    opp_parser.add_argument("--min-kelly", type=float, default=0.1, help="Min Kelly in percent")
    opp_parser.add_argument("--platform", choices=["kalshi", "polymarket"])
    opp_parser.add_argument("--limit", type=int, default=20)
    opp_parser.set_defaults(func=cmd_opportunities)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export opportunities to JSON")
    export_parser.add_argument("--output", "-o", help="Output file (stdout if not specified)")
    export_parser.add_argument("--min-edge", type=float, default=2)
    export_parser.add_argument("--min-ev", type=float, default=1.05)
    export_parser.add_argument("--min-quantity", type=int, default=100)
    export_parser.set_defaults(func=cmd_export)

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start web server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    serve_parser.set_defaults(func=cmd_serve)

    # Tags command
    tags_parser = subparsers.add_parser("tags", help="List/search Polymarket tags")
    tags_parser.add_argument("--search", "-s", help="Search for tags containing this text")
    tags_parser.add_argument("--limit", type=int, default=50)
    tags_parser.set_defaults(func=cmd_tags)

    args = parser.parse_args()
    if args.command:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
