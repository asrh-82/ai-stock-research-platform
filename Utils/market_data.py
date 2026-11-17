"""Explicit data acquisition with metadata; no sample-price fallback."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import hashlib
import importlib.metadata
import pandas as pd
import yfinance as yf
from Utils.backtesting import prepare_prices


def fetch_daily_history(ticker, start, end):
    """End is exclusive. Exclude today's incomplete session at the caller."""
    instrument = yf.Ticker(ticker)
    history = instrument.history(start=start, end=end, interval='1d', auto_adjust=False,
                                 back_adjust=False, actions=True, repair=False, keepna=True)
    if history.empty:
        raise ValueError(f'No price history returned for {ticker}. No backtest was generated.')
    metadata = instrument.get_history_metadata()
    if metadata.get('symbol', '').upper() != ticker.upper():
        raise ValueError(f'Provider identity mismatch for {ticker}.')
    if not metadata.get('currency'):
        raise ValueError(f'Provider omitted currency for {ticker}.')
    # Keep exchange-local session dates, avoiding DST offset parsing ambiguity in CSVs.
    if history.index.tz is not None:
        history.index = history.index.tz_localize(None)
    history.index = history.index.normalize()
    prepare_prices(history, ticker)
    if history.index[-1] >= pd.Timestamp(end):
        raise ValueError('Provider returned a bar outside the exclusive end date.')
    if (pd.Timestamp(end) - history.index[-1]).days > 5:
        raise ValueError(f'{ticker} history is stale: last session {history.index[-1].date()}.')
    manifest = {'provider': 'Yahoo Finance via yfinance', 'ticker': ticker.upper(),
                'currency': metadata['currency'], 'exchange': metadata.get('exchangeName'),
                'timezone': metadata.get('exchangeTimezoneName'), 'start_requested': str(start),
                'end_exclusive': str(end), 'first_session': str(history.index[0].date()),
                'last_session': str(history.index[-1].date()), 'rows': len(history),
                'retrieved_at_utc': datetime.now(ZoneInfo('UTC')).isoformat(),
                'yfinance_version': importlib.metadata.version('yfinance'),
                'price_basis': 'Provider split-adjusted OHLC and dividend-adjusted Adj Close; engine adjusts all OHLC by Adj Close / Close',
                'sha256_csv': hashlib.sha256(history.to_csv().encode()).hexdigest()}
    return history, manifest


def history_for_period(ticker, period='5y'):
    today = datetime.now(ZoneInfo('America/Toronto')).date()
    years = {'1y': 1, '3y': 3, '5y': 5, '10y': 10, 'max': 30}[period]
    start = today - timedelta(days=int(years * 365.25))
    return fetch_daily_history(ticker, start.isoformat(), today.isoformat())
