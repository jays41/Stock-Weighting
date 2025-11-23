
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


def calculate_beta(stocktckr, startperiod, endperiod, market_data_path, stock_data_path):
    market=pd.read_csv(market_data_path)
    market['Date'] = pd.to_datetime(market['Date'])

    stock=pd.read_csv(stock_data_path)
    print(stock.head())
    stock['date'] = pd.to_datetime(stock['date'])
    print(market.head())
    stock.rename(columns={"close": "Close"}, inplace=True)
    stock.rename(columns={"date": "Date"}, inplace=True)
    stock_tickr=stock[(stock["ticker"]==stocktckr)&(stock["Date"]>=startperiod)&(stock["Date"]<=endperiod)]

    stock = stock.sort_values("Date")
    market = market.sort_values("Date")

    merged=pd.merge(stock_tickr[["Date","Close"]],market[["Date","Close"]],on="Date",suffixes=("_stock","_market"))
    merged = merged.sort_values("Date")

    print(merged)
    merged["stockreturn"]=merged["Close_stock"].pct_change()
    merged["marketreturn"]=merged["Close_market"].pct_change()

    corr = merged["stockreturn"].corr(merged["marketreturn"])
    print("Correlation (daily returns):", corr)
    cov = merged["stockreturn"].cov(merged["marketreturn"])
    print("Covariance:", cov)
    varmarket = merged["marketreturn"].var()

    beta = cov / varmarket
    r2 = corr ** 2

    print("beta:",beta)
    print("R^2",r2)

    mean_stock = merged["stockreturn"].mean()
    mean_market   = merged["marketreturn"].mean()


    vol_stock = merged["stockreturn"].std()
    vol_market   = merged["marketreturn"].std()

    print("mean stock return:",mean_stock)
    print("mean market return:",mean_market)
    print("volatility stock:",vol_stock)
    print("volatility market:",vol_market)
    corr_table = merged[["stockreturn", "marketreturn"]].corr()

    print("Correlation Summary Table:")
    print(corr_table)


    sns.regplot(
        x="marketreturn",
        y="stockreturn",
        data=merged,
        scatter_kws={"alpha":0.6},
        line_kws={"color":"red"},
    )
    plt.title(f"{stocktckr} vs Market — Daily Returns")
    plt.xlabel("Market Return")
    plt.ylabel(f"{stocktckr} Return")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return beta


if __name__ == "__main__":
    stocktckr = input("input stock ticker")
    startperiod= input("input start of time period")
    endperiod=input("input end of time period")
    market_data_path = 'data\s&p_data.csv'
    stock_data_path = "data\stock_prices.csv"
    
    beta = calculate_beta(stocktckr, startperiod, endperiod, market_data_path, stock_data_path)
    print(beta)