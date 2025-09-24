import os
import asyncio
import logging
import aiohttp

import pandas as pd

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import yfinance as yf
from dotenv import load_dotenv
from agents import (
    Agent,
    Runner,
    function_tool,
    InputGuardrail,
    GuardrailFunctionOutput,
    trace,
)

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------

load_dotenv()
FINANCIAL_MODELING_PREP_API_KEY = os.getenv("FINANCIAL_MODELING_PREP_API_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not (FINANCIAL_MODELING_PREP_API_KEY and FINNHUB_API_KEY and OPENAI_API_KEY):
    raise EnvironmentError("Missing one or more required API keys.")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------

BASE_FMP_URL = "https://financialmodelingprep.com/api/v3"
BASE_FINNHUB_URL = "https://finnhub.io/api/v1"


async def fetch_json(session: aiohttp.ClientSession, url: str) -> Optional[Any]:
    try:
        async with session.get(url, timeout=15) as response:
            response.raise_for_status()
            return await response.json()
    except Exception as e:
        logger.error("Request failed for %s: %s", url, e)
        return None

async def get_historical_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Fetch historical stock data (default 1 year) using yfinance."""
    loop = asyncio.get_event_loop()
    df: pd.DataFrame = await loop.run_in_executor(None, lambda: yf.Ticker(symbol).history(period=period))
    return df

# ---------------------------------------------------------------------
# In-memory async-safe cache
# ---------------------------------------------------------------------

class AsyncCache:
    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self._store: Dict[str, Tuple[float, Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            value = self._store.get(key)
            if not value:
                return None
            timestamp, data = value
            if (datetime.now().timestamp() - timestamp) > self.ttl:
                self._store.pop(key, None)
                return None
            return data

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            self._store[key] = (datetime.now().timestamp(), value)


cache = AsyncCache(ttl_seconds=600)  # cache for 10 minutes


# ---------------------------------------------------------------------
# Data fetch functions
# ---------------------------------------------------------------------

async def get_stock_price(session: aiohttp.ClientSession, symbol: str) -> Dict[str, Any]:
    url = f"{BASE_FMP_URL}/quote-order/{symbol}?apikey={FINANCIAL_MODELING_PREP_API_KEY}"
    data = await fetch_json(session, url)
    if not data:
        return {"error": f"Could not fetch price for {symbol}"}
    try:
        record = data[0]
        return {
            "symbol": symbol.upper(),
            "price": record["price"],
            "volume": record["volume"],
            "priceAvg50": record["priceAvg50"],
            "priceAvg200": record["priceAvg200"],
            "EPS": record["eps"],
            "PE": record["pe"],
            "earningsAnnouncement": record["earningsAnnouncement"],
        }
    except (IndexError, KeyError):
        return {"error": f"Malformed stock data for {symbol}"}


async def get_company_financials(session: aiohttp.ClientSession, symbol: str) -> Dict[str, Any]:
    url = f"{BASE_FMP_URL}/profile/{symbol}?apikey={FINANCIAL_MODELING_PREP_API_KEY}"
    data = await fetch_json(session, url)
    if not data:
        return {"error": f"Could not fetch financials for {symbol}"}
    try:
        record = data[0]
        return {
            "symbol": record["symbol"],
            "companyName": record["companyName"],
            "marketCap": record["mktCap"],
            "industry": record["industry"],
            "sector": record["sector"],
            "website": record["website"],
            "beta": record["beta"],
            "price": record["price"],
        }
    except (IndexError, KeyError):
        return {"error": f"Malformed financials for {symbol}"}


async def get_income_statement(session: aiohttp.ClientSession, symbol: str) -> Dict[str, Any]:
    url = f"{BASE_FMP_URL}/income-statement/{symbol}?period=annual&apikey={FINANCIAL_MODELING_PREP_API_KEY}"
    data = await fetch_json(session, url)
    if not data:
        return {"error": f"Could not fetch income statement for {symbol}"}
    try:
        record = data[0]
        return {
            "date": record["date"],
            "revenue": record["revenue"],
            "grossProfit": record["grossProfit"],
            "netIncome": record["netIncome"],
            "ebitda": record["ebitda"],
            "EPS": record["eps"],
            "EPS diluted": record["epsdiluted"],
        }
    except (IndexError, KeyError):
        return {"error": f"Malformed income statement for {symbol}"}


async def get_company_news(session: aiohttp.ClientSession, symbol: str, max_results: int = 10, days: int = 3) -> List[Dict[str, Any]]:
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")
    url = (
        f"{BASE_FINNHUB_URL}/company-news?symbol={symbol}"
        f"&from={start}&to={end}&token={FINNHUB_API_KEY}"
    )
    news = await fetch_json(session, url)
    if not news:
        return []
    return news[:max_results]


# ---------------------------------------------------------------------
# Concurrency wrapper with caching
# ---------------------------------------------------------------------

async def fetch_all_company_data(symbol: str) -> Dict[str, Any]:
    # check cache
    cached = await cache.get(symbol)
    if cached:
        logger.info("Cache hit for %s", symbol)
        return cached

    async with aiohttp.ClientSession() as session:
        stock_price_task = get_stock_price(session, symbol)
        financials_task = get_company_financials(session, symbol)
        income_task = get_income_statement(session, symbol)
        news_task = get_company_news(session, symbol)

        stock_price, financials, income, news = await asyncio.gather(
            stock_price_task, financials_task, income_task, news_task
        )

    result = {
        "stock_price": stock_price,
        "financials": financials,
        "income_statement": income,
        "news": news,
    }

    # store in cache
    await cache.set(symbol, result)
    return result


# ---------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------

@function_tool
async def get_company_data_tool(symbol: str) -> Dict[str, Any]:
    return await fetch_all_company_data(symbol)

@function_tool
async def get_historical_data_tool(symbol: str, period: str = "1y") -> Dict[str, Any]:
    df = await get_historical_data(symbol, period)
    return df.reset_index().to_dict(orient="records")  # JSON safe for LLM

# ---------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------

async def finance_input_guardrail(ctx, agent, input_data: str) -> GuardrailFunctionOutput:
    keywords = ["finance", "investment", "stock", "market", "revenue", "company", "price"]
    is_finance_query = any(k in input_data.lower() for k in keywords)
    return GuardrailFunctionOutput(
        output_info={"is_finance_query": is_finance_query},
        tripwire_triggered=not is_finance_query,
    )


# ---------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------

data_agent = Agent(
    name="Data Agent",
    instructions="Fetch stock price, financials, income statement, and news concurrently using the tool.",
    tools=[get_company_data_tool],
    model="gpt-5-nano",
)

analysis_agent = Agent(
    name="Analysis Agent",
    instructions=(
        "Analyze the provided company data. Summarize:\n"
        "1. Key financial metrics\n"
        "2. News highlights\n"
        "3. Comparative insights and investor recommendations\n"
        "Output as HTML."
    ),
    model="gpt-5-nano",
)

investment_thesis_agent = Agent(
    name="Investment Thesis Agent",
    instructions=(
        "Write a detailed investment thesis in HTML format using the data provided. "
        "Provide numeric justification, a clear buy/hold/sell recommendation, "
        "and analyze short-term vs long-term investment potential. Reference all sources."
    ),
    model="gpt-5-nano",
)

coordinator_agent = Agent(
    name="Coordinator",
    instructions=(
        "You are the coordinator. "
        "Your job is to run exactly two steps in sequence:\n"
        "1. Call the Investment Thesis Agent to create a detailed investment thesis.\n"
        "2. Call the Analysis Agent to produce financial analysis.\n\n"
        "Do not request any additional tools or propose further actions. "
        "Always finalize after both agents have run."
    ),
    model="gpt-5-nano",
)


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

async def main(stock_symbol: str, period: str = "1y") -> Tuple[str, pd.DataFrame]:
    # Prepare raw query
    query = (
        f"Provide the latest stock price, company financials, income statement, "
        f"recent news, and generate an investment thesis for {stock_symbol}."
    )

    # Fetch supporting data
    fetched_data = await fetch_all_company_data(stock_symbol)
    hist_df = await get_historical_data(stock_symbol, period=period)

    # Construct base conversation
    conversation = [
        {"role": "system", "content": f"Fetched data for {stock_symbol}: {fetched_data}"},
        {"role": "system", "content": f"Historical data summary (period={period}): {hist_df.describe().to_dict()}"},
        {"role": "user", "content": query},
    ]

    # Coordinator orchestrates both steps
    with trace(workflow_name="Financial_Data_Workflow", group_id="finance_group"):
        result = await Runner.run(coordinator_agent, conversation, max_turns=3)

    final_output = result.final_output

    print("Final Analysis Report (HTML):")
    print(final_output)

    return final_output, hist_df

def run_analysis(stock_symbol: str, period: str = "1y") -> Tuple[str, pd.DataFrame]:
    return asyncio.run(main(stock_symbol, period))