import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

import yfinance as yf
from yahooquery import Ticker
from openai import OpenAI
import finnhub
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OpenAI.api_key = OPENAI_API_KEY
client = OpenAI()

DATA_FILE = Path("data/investment.txt")


class Stock(BaseModel):
    company_name: str
    company_ticker: str


def get_company_news(ticker: str, days: int = 7, limit: int = 10) -> List[dict]:
    """Fetch latest company news from Finnhub."""
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    news = finnhub_client.company_news(
        symbol=ticker,
        _from=start_date.strftime("%Y-%m-%d"),
        to=end_date.strftime("%Y-%m-%d"),
    )
    return news[:limit]


def write_news_to_file(news: List[dict], filename: Path) -> None:
    """Write news headlines to a file."""
    with filename.open("w", encoding="utf-8") as file:
        for item in news:
            file.write(
                f"Title: {item.get('headline', 'No title')}\n"
                f"Link: {item.get('url', 'No link')}\n"
                f"Date: {item.get('datetime', 'No date')}\n\n"
            )


def append_to_file(content: str, filename: Path = DATA_FILE) -> None:
    """Append content to the investment file."""
    with filename.open("a", encoding="utf-8") as file:
        file.write(content + "\n")


def get_stock_evolution(ticker: str, period: str = "1y") -> Optional[str]:
    """Fetch historical stock data and append to file."""
    try:
        hist = yf.Ticker(ticker).history(period=period)
    except Exception as e:
        print(f"Error fetching stock history for {ticker}: {e}")
        return None

    append_to_file(f"Stock Evolution for {ticker}:\n{hist.to_string()}")
    return hist


def get_financial_statements(ticker: str) -> None:
    """Fetch and append financial statements to the investment file."""
    company = Ticker(ticker)

    statements = {
        "Balance Sheet": getattr(company.balance_sheet(), "to_string", lambda: str(company.balance_sheet()))(),
        "Cash Flow": getattr(company.cash_flow(trailing=False), "to_string", lambda: str(company.cash_flow(trailing=False)))(),
        "Income Statement": getattr(company.income_statement(), "to_string", lambda: str(company.income_statement()))(),
    }

    try:
        valuation_measures = str(company.valuation_measures)
    except Exception:
        valuation_measures = "N/A"

    statements["Valuation Measures"] = valuation_measures

    for section, content in statements.items():
        append_to_file(f"\n{section}\n{content}")


def get_data(stock: Stock, period: str = "1y", filename: Path = DATA_FILE) -> Optional[str]:
    """Fetch stock history, financial statements, and news."""
    hist = get_stock_evolution(stock.company_ticker, period)

    get_financial_statements(stock.company_ticker)

    news = get_company_news(stock.company_ticker)
    if news:
        write_news_to_file(news, filename)
    else:
        print("No news found.")

    return hist


def run_analysis(request: str) -> tuple[str, Optional[str]]:
    """Run full investment analysis based on a user request."""
    print(f"Received request: {request}")

    # Step 1: Extract company name and ticker using OpenAI
    response = client.responses.create(
        model="gpt-5-nano",
        input=[{
            "role": "user",
            "content": (
                f"Given the user request, extract the company name and stock ticker: {request}. "
                "Return in JSON format as {\"company_name\": ..., \"company_ticker\": ...}"
            )
        }],
    )

    output_text = "".join(
        content.text
        for item in response.output
        if getattr(item, "type", None) == "message"
        for content in getattr(item, "content", [])
        if hasattr(content, "text")
    )

    params = json.loads(output_text)
    stock = Stock(**params)

    hist_data = get_data(stock)

    # Step 2: Read accumulated investment data
    content = DATA_FILE.read_text(encoding="utf-8")[:14000]

    # Step 3: Generate investment thesis
    response = client.responses.create(
        model="gpt-5-nano",
        input=[
            {"role": "user", "content": request},
            {"role": "system", "content": (
                "Write a detailed investment thesis in HTML format using the data provided. "
                "Provide numeric justification, a clear buy/hold/sell recommendation, "
                "and analyze short-term vs long-term investment potential. Reference all sources."
            )},
            {"role": "assistant", "content": content},
        ],
    )

    thesis = "".join(
        content.text
        for item in response.output
        if getattr(item, "type", None) == "message"
        for content in getattr(item, "content", [])
        if hasattr(content, "text")
    )

    return thesis, hist_data
