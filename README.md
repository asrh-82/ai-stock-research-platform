# AI Stock Research Platform

A Streamlit-based market research dashboard for stock analysis, peer comparison, watchlist management, and portfolio tracking.

## Features

- Company and ticker search
- Live market data through yfinance
- Interactive price charts
- Rule-based stock research scoring
- Fundamental analysis dashboard
- Risk metrics including volatility, max drawdown, and Sharpe ratio
- Multi-stock comparison
- Persistent watchlist management
- Persistent portfolio tracking
- Portfolio gain/loss analysis
- Portfolio allocation and sector exposure

## Tech Stack

- Python
- Streamlit
- yfinance
- pandas
- numpy
- plotly
- uv

## Project Structure

```text
ai-stock-research-platform/
├── app.py
├── README.md
├── pyproject.toml
├── uv.lock
│
├── Utils/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── portfolio_utils.py
│   ├── scoring.py
│   └── ui_sections.py
│
└── Data/
    ├── portfolio.json
    └── watchlist.json
```

## Running the App

```bash
uv run streamlit run app.py
```

## File Responsibilities

### app.py

- Application entry point
- Streamlit initialization
- Tab routing
- High-level application flow

### Utils/data_utils.py

- Company search
- Market data retrieval
- Price history retrieval
- Financial statement retrieval
- Earnings and recommendation retrieval
- Comparison dataset generation

### Utils/portfolio_utils.py

- Portfolio persistence
- Watchlist persistence
- Portfolio calculations
- Portfolio summaries
- Portfolio highlights
- Portfolio and watchlist CRUD operations

### Utils/scoring.py

- Return calculations
- Volatility calculations
- Maximum drawdown calculations
- Sharpe ratio calculations
- Revenue growth calculations
- Research score generation

### Utils/ui_sections.py

- Styling
- Session state initialization
- Dashboard rendering
- Analysis rendering
- Comparison rendering
- Watchlist rendering
- Portfolio rendering

## Roadmap

### Completed

- Stock research dashboard
- Company search
- Watchlist management
- Portfolio tracking
- Company comparison tools
- Research scoring system
- Modular codebase refactor

### Planned

- Portfolio benchmarking
- DCF valuation model
- Enhanced analytics
- Earnings calendar
- AI-generated research summaries

## Disclaimer

This project is for educational purposes only.

Nothing in this application constitutes financial or investment advice.