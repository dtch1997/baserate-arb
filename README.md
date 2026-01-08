# Base Rate Arbitrage Scanner

Find mispriced prediction markets by comparing market prices to historical base rates researched by an LLM agent.

## The Core Idea

Prediction markets are often mispriced because:
1. Participants anchor on recent events or availability bias
2. Low-probability events are systematically overpriced
3. Time-decay of probability isn't properly accounted for

This tool uses an LLM agent to research historical base rates (how often events actually occur), then compares those to market prices to find edges.

## How It Works

### 1. Fetch Markets
Pull active markets from Kalshi and/or Polymarket, including:
- Market title and description
- Resolution criteria (how it gets resolved)
- Resolution date
- Current prices and order book depth

### 2. Research Base Rates (LLM Agent)
The agent (Claude) researches each market to find:
- **The base rate**: Historical probability of this type of event
- **The unit**: per-year, per-month, per-event, or absolute (one-time)
- **Events per period**: For per-event rates (e.g., 50 press conferences/year)
- **Confidence**: How certain the agent is (0-1)
- **Sources**: Where the data came from

### 3. Time-Adjusted Fair Probability
For time-based rates, probability compounds over time:

```
P(event occurs before resolution) = 1 - (1 - rate)^periods_remaining
```

Example: A 10% annual rate of hurricanes hitting Florida:
- 1 year out: 10% probability
- 6 months out: ~5% probability
- 1 month out: ~0.8% probability

### 4. Find Edge & Calculate EV
- **Edge** = Fair Probability - Market Probability
- **EV** = (Fair Prob × 100) / Buy Price
- If EV > 1, the bet has positive expected value

### 5. Kelly Criterion Position Sizing
The Kelly formula tells you optimal bet size:
```
f* = (bp - q) / b
```
Where b = odds, p = win prob, q = lose prob

We recommend using **half Kelly** (0.5x) for safety.

### 6. Order Book Integration
Instead of just using the displayed price, we look at the order book to find:
- Price at which you can actually buy ~1000+ contracts
- Available liquidity at that price
- This gives more realistic fill prices

## Installation

```bash
git clone https://github.com/katherineworden/baserate-arb.git
cd baserate-arb

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Add your ANTHROPIC_API_KEY (required)
# Optionally add KALSHI_API_KEY, ODDS_API_KEY, etc.
```

## Usage

### Web UI
```bash
python run.py
# Open http://127.0.0.1:8000
```

The UI has:
- **Fetch buttons**: Pull markets from Kalshi/Polymarket
- **Update Base Rates**: Run LLM agent on markets without base rates
- **Refresh Table**: Recalculate opportunities with current prices
- **Sortable columns**: Click headers to sort by EV, edge, Kelly, etc.
- **Filters**: Min edge, min EV, min quantity, Kelly range, platform

### CLI
```bash
# Fetch markets from all platforms
python cli.py fetch --platform all --limit 100

# Research base rates (uses Claude API)
python cli.py research --limit 20

# Show opportunities
python cli.py opportunities --min-edge 3 --min-ev 1.1

# Export to JSON
python cli.py export -o opportunities.json
```

## Understanding the Output

| Column | Meaning |
|--------|---------|
| Market | The prediction market title |
| Platform | kalshi or polymarket |
| Side | YES or NO (which side to buy) |
| Fair % | Base rate probability adjusted for time |
| Market % | Current market implied probability |
| Edge | Fair - Market (positive = underpriced) |
| EV | Expected value multiplier (>1 = +EV) |
| Kelly % | Recommended bankroll fraction |
| Price | Limit order price in cents |
| Qty | Available contracts at that price |

## Filters Explained

### Min Edge
Minimum percentage point edge required. Default 2%.

### Min EV
Minimum expected value multiplier. Default 1.05 (5% expected profit).

### Min Edge Ratio (Significance Filter)
Edge must be at least X% of fair probability. This filters out noise like:
- Fair = 2%, Market = 1% → edge = 1%
- But 1%/2% = 50% ratio, which might just be estimation error

Set `min_edge_ratio=0.5` to require edge ≥ 50% of fair prob.

### Min Confidence
Only show opportunities where the LLM agent has confidence ≥ X in its base rate estimate.

### Kelly Range
Filter by recommended bet size. High Kelly (>25%) might indicate model overconfidence.

## Base Rate Units

| Unit | When to Use | Example |
|------|-------------|---------|
| `per_year` | Annual occurrence rates | Hurricanes, elections |
| `per_month` | Monthly rates | Fed rate decisions |
| `per_week` | Weekly rates | Sports outcomes |
| `per_day` | Daily rates | Weather events |
| `per_event` | Per specific event type | Per press conference, per game |
| `absolute` | One-time events | "Will X happen before 2025?" |

For `per_event`, also specify `events_per_period` (events per year).

## Architecture

```
src/
├── models/
│   └── market.py       # Market, BaseRate, OrderBook, OpportunityAnalysis
├── clients/
│   ├── kalshi.py       # Kalshi API client
│   ├── polymarket.py   # Polymarket CLOB client
│   └── odds_api.py     # Sportsbook odds (for reference)
├── agents/
│   └── base_rate_agent.py  # Claude-powered research agent
├── web/
│   ├── app.py          # FastAPI server
│   └── templates/      # Web UI
├── storage.py          # JSON persistence for markets/rates
└── analyzer.py         # Opportunity detection logic
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/opportunities` | GET | Get filtered opportunities |
| `/api/markets` | GET | Get stored markets |
| `/api/fetch/kalshi` | POST | Fetch Kalshi markets |
| `/api/fetch/polymarket` | POST | Fetch Polymarket markets |
| `/api/research/base_rate/{id}` | POST | Research single market |
| `/api/research/batch` | POST | Research multiple markets |
| `/api/portfolio/kelly` | POST | Calculate Kelly portfolio |
| `/api/sportsbook/odds` | GET | Get sportsbook reference odds |

## Configuration

### Required
- `ANTHROPIC_API_KEY` - For the base rate research agent

### Optional
- `KALSHI_API_KEY` + `KALSHI_PRIVATE_KEY_PATH` - For authenticated Kalshi access
- `POLYMARKET_API_KEY` - Polymarket public API works without auth
- `ODDS_API_KEY` - The Odds API for sportsbook reference

## Example Workflow

1. **Fetch markets**: Click "Fetch Kalshi" and "Fetch Polymarket"
2. **Research base rates**: Click "Update Base Rates (LLM)" - this runs Claude on each market
3. **Review opportunities**: Sort by EV, filter out low-confidence or low-edge
4. **Check reasoning**: Click a market to see the agent's reasoning
5. **Trade**: Use the recommended price and quantity as a starting point

## Limitations & Caveats

1. **Base rate quality depends on available data** - The LLM can only work with what it knows
2. **Low-probability events are hard** - Hard to distinguish 0.1% from 1%
3. **Reference class problems** - "Will X happen?" depends heavily on how you define X
4. **Market prices may be right** - The market might know something the base rate doesn't
5. **Liquidity matters** - Small markets may not have enough depth to trade

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_models.py -v
python -m pytest tests/test_analyzer.py -v
python -m pytest tests/test_security.py -v
```

## License

MIT

## Disclaimer

This is not financial advice. Use at your own risk. Past base rates don't guarantee future results.
