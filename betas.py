
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


def calculate_beta_single(stocktckr, startperiod, endperiod, market_data, stock_data):
    stock_tickr = stock_data[
        (stock_data["ticker"] == stocktckr) & 
        (stock_data["Date"] >= startperiod) & 
        (stock_data["Date"] <= endperiod)
    ].copy()
    
    if len(stock_tickr) == 0:
        print(f"Warning: No data found for ticker {stocktckr} in the specified date range")
        return np.nan
    
    merged = pd.merge(
        stock_tickr[["Date", "Close"]], 
        market_data[["Date", "Close"]], 
        on="Date", 
        suffixes=("_stock", "_market")
    ).sort_values("Date")
    
    if len(merged) < 2:
        print(f"Warning: Insufficient data for ticker {stocktckr} after merging")
        return np.nan
    
    merged["stockreturn"] = merged["Close_stock"].pct_change()
    merged["marketreturn"] = merged["Close_market"].pct_change()
    
    # Remove NaN values
    merged = merged.dropna()
    
    if len(merged) < 2:
        print(f"Warning: Insufficient return data for ticker {stocktckr}")
        return np.nan
    
    # Calculate beta
    cov = merged["stockreturn"].cov(merged["marketreturn"])
    varmarket = merged["marketreturn"].var()
    
    if varmarket == 0 or np.isnan(varmarket):
        print(f"Warning: Market variance is zero or NaN for ticker {stocktckr}")
        return np.nan
    
    beta = cov / varmarket
    
    return beta


def calculate_beta(tickers, startperiod, endperiod, market_data_path, stock_data_path):
    market = pd.read_csv(market_data_path)
    market['Date'] = pd.to_datetime(market['Date'])
    
    stock = pd.read_csv(stock_data_path)
    stock['Date'] = pd.to_datetime(stock['date'])
    stock.rename(columns={"close": "Close"}, inplace=True)
    
    stock = stock.sort_values("Date")
    market = market.sort_values("Date")
    
    # Convert dates if they're strings
    if isinstance(startperiod, str):
        startperiod = pd.to_datetime(startperiod)
    if isinstance(endperiod, str):
        endperiod = pd.to_datetime(endperiod)
    
    betas = {}
    for ticker in tickers:
        beta = calculate_beta_single(ticker, startperiod, endperiod, market, stock)
        betas[ticker] = beta
        
    return pd.Series(betas, index=tickers)