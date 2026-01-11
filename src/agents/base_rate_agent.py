"""Base rate research agent using Anthropic Claude."""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import anthropic
import httpx

from src.models.market import BaseRate, BaseRateUnit, Market


@dataclass
class ToolCall:
    """Record of a single tool call."""
    name: str
    input: dict[str, Any]
    output: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Iteration:
    """Record of a single agent iteration."""
    index: int
    tool_calls: list[ToolCall] = field(default_factory=list)
    assistant_text: Optional[str] = None
    stop_reason: Optional[str] = None


@dataclass
class ResearchTrace:
    """Full trace of a research_base_rate call."""
    market_id: str
    market_title: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    iterations: list[Iteration] = field(default_factory=list)
    result: Optional[BaseRate] = None
    error: Optional[str] = None

    @property
    def total_tool_calls(self) -> int:
        return sum(len(it.tool_calls) for it in self.iterations)

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> dict:
        """Convert trace to dictionary for serialization."""
        return {
            "market_id": self.market_id,
            "market_title": self.market_title,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "total_iterations": len(self.iterations),
            "total_tool_calls": self.total_tool_calls,
            "iterations": [
                {
                    "index": it.index,
                    "stop_reason": it.stop_reason,
                    "assistant_text": it.assistant_text,
                    "tool_calls": [
                        {"name": tc.name, "input": tc.input, "output": tc.output}
                        for tc in it.tool_calls
                    ]
                }
                for it in self.iterations
            ],
            "result": {
                "rate": self.result.rate,
                "unit": self.result.unit.value,
                "confidence": self.result.confidence,
                "reasoning": self.result.reasoning,
            } if self.result else None,
            "error": self.error
        }


# Tool definitions for the agent
TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web for information about historical base rates, statistics, and reference data. Use this to find data on how often events occur.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant statistics and base rate information"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate_base_rate",
        "description": "Calculate and store the base rate for a market after gathering information. Call this when you have enough information to determine the base rate.",
        "input_schema": {
            "type": "object",
            "properties": {
                "rate": {
                    "type": "number",
                    "description": "The base rate probability (0 to 1). For per-period rates, this is the probability per period."
                },
                "unit": {
                    "type": "string",
                    "enum": ["per_year", "per_month", "per_week", "per_day", "per_event", "absolute"],
                    "description": "The unit of the base rate. Use 'absolute' for one-time events, 'per_year' for annual rates, etc."
                },
                "reasoning": {
                    "type": "string",
                    "description": "Detailed explanation of how you calculated this base rate, including sources and methodology."
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of sources/URLs used to determine the base rate"
                },
                "events_per_period": {
                    "type": "integer",
                    "description": "For per_event unit: estimated number of events in the time period. E.g., if there are ~50 press conferences per year, put 50."
                },
                "confidence": {
                    "type": "number",
                    "description": "Your confidence in this base rate (0 to 1). Use 0.8+ if you have solid historical data, 0.5-0.7 if extrapolating, 0.3-0.5 if highly uncertain."
                }
            },
            "required": ["rate", "unit", "reasoning", "confidence"]
        }
    }
]


class BaseRateAgent:
    """
    Agent that researches and calculates base rates for prediction markets.

    Uses Claude with tool use to:
    1. Understand the market and resolution criteria
    2. Search for relevant historical data
    3. Calculate an appropriate base rate with proper units
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self._http = httpx.Client(timeout=30.0)

    def _execute_web_search(self, query: str) -> str:
        """
        Execute a web search using a search API.

        For now, this uses a simple approach. In production, you'd want
        to use a proper search API (Tavily, Serper, etc.)
        """
        # Try using DuckDuckGo instant answers as a fallback
        try:
            response = self._http.get(
                "https://api.duckduckgo.com/",
                params={
                    "q": query,
                    "format": "json",
                    "no_html": 1,
                    "skip_disambig": 1
                }
            )
            data = response.json()

            results = []
            if data.get("Abstract"):
                results.append(f"Summary: {data['Abstract']}")
                if data.get("AbstractSource"):
                    results.append(f"Source: {data['AbstractSource']}")

            if data.get("RelatedTopics"):
                for topic in data["RelatedTopics"][:3]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append(f"- {topic['Text']}")

            if results:
                return "\n".join(results)
            else:
                return f"No direct results found for '{query}'. Consider trying a more specific query or using known statistics."

        except Exception as e:
            return f"Search failed: {str(e)}. Please estimate based on general knowledge or try a different query."

    def _process_tool_call(self, tool_name: str, tool_input: dict) -> tuple[str, Optional[BaseRate]]:
        """Process a tool call and return result."""
        if tool_name == "web_search":
            result = self._execute_web_search(tool_input["query"])
            return result, None

        elif tool_name == "calculate_base_rate":
            base_rate = BaseRate(
                rate=tool_input["rate"],
                unit=BaseRateUnit(tool_input["unit"]),
                reasoning=tool_input["reasoning"],
                sources=tool_input.get("sources", []),
                events_per_period=tool_input.get("events_per_period"),
                confidence=tool_input.get("confidence", 0.5),
                last_updated=datetime.utcnow()
            )
            return "Base rate calculated and stored.", base_rate

        return "Unknown tool", None

    def research_base_rate(
        self,
        market: Market,
        max_iterations: int = 5
    ) -> tuple[Optional[BaseRate], ResearchTrace]:
        """
        Research and calculate base rate for a market.

        The agent will:
        1. Analyze the market question and resolution criteria
        2. Search for relevant historical data
        3. Calculate an appropriate base rate

        Args:
            market: The market to research
            max_iterations: Maximum tool use iterations

        Returns:
            Tuple of (BaseRate if successfully calculated, ResearchTrace with full execution details)
        """
        trace = ResearchTrace(market_id=market.id, market_title=market.title)
        system_prompt = """You are a base rate research agent for prediction markets. Your job is to find historical base rates for events.

IMPORTANT GUIDELINES:
1. UNITS MATTER: Determine if the base rate should be:
   - per_year: For things that happen X times per year (e.g., hurricanes, elections)
   - per_month: For monthly occurrences
   - per_week: For weekly occurrences
   - per_day: For daily occurrences
   - per_event: For things that happen per specific event type (e.g., per press conference, per game)
   - absolute: For one-time events with a fixed probability

2. For per_event rates, estimate how many such events occur in the relevant time period.

3. Be conservative - it's better to be uncertain than overconfident.

4. Consider selection effects and reference class issues.

5. Look for multiple sources when possible.

6. If you can't find good data, make a reasoned estimate and clearly state your uncertainty.

7. IMPORTANT: Do not call web_search more than 2 times. If search results are unhelpful, use your general knowledge to estimate.

After gathering information (or if searches are unsuccessful), ALWAYS call calculate_base_rate with your best estimate."""

        user_message = f"""Please research the base rate for this prediction market:

MARKET TITLE: {market.title}

DESCRIPTION: {market.description}

RESOLUTION CRITERIA: {market.resolution_criteria}

RESOLUTION DATE: {market.resolution_date.strftime('%Y-%m-%d')}

CATEGORY: {market.category}

Research this market and calculate an appropriate base rate. Consider:
1. What is the reference class for this event?
2. How often do similar events occur historically?
3. What is the appropriate time unit for the base rate?
4. Are there any special circumstances that might affect the probability?

Use web_search to find relevant data, then call calculate_base_rate with your findings."""

        messages = [{"role": "user", "content": user_message}]

        base_rate = None
        iteration_count = 0

        try:
            while iteration_count < max_iterations:
                current_iteration = Iteration(index=iteration_count)

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system_prompt,
                    tools=TOOLS,
                    messages=messages
                )

                current_iteration.stop_reason = response.stop_reason

                # Extract any text from the response
                for block in response.content:
                    if hasattr(block, "text"):
                        current_iteration.assistant_text = block.text

                # Check if we got a final response
                if response.stop_reason == "end_turn":
                    trace.iterations.append(current_iteration)
                    break

                # Process tool uses
                if response.stop_reason == "tool_use":
                    # Add assistant's response
                    messages.append({
                        "role": "assistant",
                        "content": response.content
                    })

                    # Process each tool use
                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            result, calculated_rate = self._process_tool_call(
                                block.name,
                                block.input
                            )

                            # Record tool call in trace
                            current_iteration.tool_calls.append(ToolCall(
                                name=block.name,
                                input=block.input,
                                output=result
                            ))

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result
                            })

                            if calculated_rate:
                                base_rate = calculated_rate

                    messages.append({"role": "user", "content": tool_results})

                trace.iterations.append(current_iteration)
                iteration_count += 1

        except Exception as e:
            trace.error = str(e)

        trace.result = base_rate
        trace.completed_at = datetime.utcnow()
        return base_rate, trace

    def batch_research(
        self,
        markets: list[Market],
        skip_existing: bool = True
    ) -> tuple[dict[str, BaseRate], list[ResearchTrace]]:
        """
        Research base rates for multiple markets.

        Args:
            markets: List of markets to research
            skip_existing: Skip markets that already have base rates

        Returns:
            Tuple of (dict mapping market ID to BaseRate, list of all traces)
        """
        results = {}
        traces = []

        for market in markets:
            if skip_existing and market.base_rate:
                results[market.id] = market.base_rate
                continue

            base_rate, trace = self.research_base_rate(market)
            traces.append(trace)
            if base_rate:
                results[market.id] = base_rate
                market.base_rate = base_rate

        return results, traces

    def close(self):
        """Close HTTP client."""
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class EnhancedBaseRateAgent(BaseRateAgent):
    """
    Enhanced agent with additional search capabilities.

    Supports integration with better search APIs when available.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        tavily_api_key: Optional[str] = None,
        serper_api_key: Optional[str] = None
    ):
        super().__init__(api_key, model)
        self.tavily_api_key = tavily_api_key
        self.serper_api_key = serper_api_key

    def _execute_web_search(self, query: str) -> str:
        """Execute web search with enhanced APIs if available."""
        # Try Tavily first (best for research)
        if self.tavily_api_key:
            try:
                response = self._http.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self.tavily_api_key,
                        "query": query,
                        "search_depth": "advanced",
                        "include_answer": True,
                        "max_results": 5
                    }
                )
                data = response.json()

                results = []
                if data.get("answer"):
                    results.append(f"Summary: {data['answer']}")

                for result in data.get("results", [])[:5]:
                    title = result.get("title", "")
                    content = result.get("content", "")
                    url = result.get("url", "")
                    results.append(f"\n[{title}]({url})\n{content}")

                if results:
                    return "\n".join(results)
            except Exception:
                pass  # Fall through to next option

        # Try Serper (Google search)
        if self.serper_api_key:
            try:
                response = self._http.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": self.serper_api_key},
                    json={"q": query}
                )
                data = response.json()

                results = []
                if data.get("answerBox", {}).get("answer"):
                    results.append(f"Answer: {data['answerBox']['answer']}")

                for result in data.get("organic", [])[:5]:
                    title = result.get("title", "")
                    snippet = result.get("snippet", "")
                    link = result.get("link", "")
                    results.append(f"\n[{title}]({link})\n{snippet}")

                if results:
                    return "\n".join(results)
            except Exception:
                pass

        # Fall back to DuckDuckGo
        return super()._execute_web_search(query)
