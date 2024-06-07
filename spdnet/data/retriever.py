import yfinance as yf
import os
from spdnet.data.utils import get_next_date, plot_prices
import pandas as pd


def get_rolling_covariance_matrices(tickers, start_date, end_date, window_size):

    data = yf.download(tickers, start=start_date, end=end_date)
    
  
    adj_close = data['Adj Close']
    
   
    daily_returns = adj_close.pct_change().dropna()
    
   
    rolling_cov_matrices = daily_returns.rolling(window=window_size).cov(pairwise=True)
    
    return rolling_cov_matrices


def get_historical_overview(ticker_symbol, dest_path=None):
    """Given a single ticker, returns historical daily closing prices for the max available time interval
    If destination path is provided, it will save the resulting csv data on disk"""
    stock = yf.Ticker(ticker_symbol)
    data = stock.history(period="max")["Close"]
    if dest_path != None:
        start = str(data.index[0]).split()[0]
        end = str(data.index[-1]).split()[0]
        dest_path = os.path.join(
            dest_path, ticker_symbol + "_" + start + "_" + end + ".csv"
        )
        data.to_csv(dest_path, index=True)
    return data


def get_closing(ticker_symbol, start, end, interval="1h", dest_path=None):
    """Given a single ticker, it returns the highfrequency (based on the interval) closing prices between that start date and end date.
    If destination path is provided, it will save the resulting csv data on disk.
    interval can be: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    Unfortunately for Yahoo Finance "API" for high_freq interval (<=1h) data must be within last 730 days from today
    """
    stock = yf.Ticker(ticker_symbol)
    data = stock.history(start=start, end=end, interval=interval)["Close"]
    if dest_path != None:
        dest_path = os.path.join(
            dest_path, ticker_symbol + "_" + start + "_" + end + ".csv"
        )
        data.to_csv(dest_path, index=True)
    return data



def get_historical_highfreq_closing(
    ticker_symbol, start_date, end_date, interval="1h", dest_path=None
):
    """Given a single ticker, returns closing prices data frame (indexed by date-time), from start_date to end_date, calling get_closing multiple times.
    In case data are not available in the specified interval of time, will return maximum amount of available data of such interval
    """
    df = pd.DataFrame()
    i_date = start_date
    while i_date != get_next_date(end_date, 1):
        n_date = get_next_date(i_date, 1)
        hf_prices = get_closing(ticker_symbol, i_date, n_date, interval, dest_path=None)
        if df.empty:
            df = hf_prices
        else:
            df = pd.concat([df, hf_prices])
        i_date = n_date

    if dest_path != None:
        start = str(df.index[0]).split()[0]
        end = str(df.index[-1]).split()[0]
        dest_path = os.path.join(
            dest_path, ticker_symbol + "_" + start + "_" + end + ".csv"
        )
        df.to_csv(dest_path, index=True)
    return df




if __name__ == "__main__":
    #example
    stocks = ["AAPL", "MSFT"]
    start_date="2024-05-08"
    end_date= "2024-05-10"
    get_closing("AAPL", start_date, end_date, interval="1h")
    prices = pd.DataFrame()
    for stock in stocks:
        if prices.empty:
            prices = get_historical_highfreq_closing(stock, start_date, end_date, interval="1h")
        else:
            prices = pd.merge(
                prices,
                get_historical_highfreq_closing(stock, start_date, end_date, interval="1h"),
                how="inner",
                left_index=True,
                right_index=True,
            )
    prices.columns = stocks
    
    plot_prices(prices)
    