import pandas as pd
import math
from spdnet.data.utils import get_next_date, plot_histograms
from spdnet.data.retriever import get_closing
import os 



def get_simple_return(p_curr, p_prec):
    """Given p_curr (P_t) and p_prec (P_{t-1}) returns the simple return
    
    Args:
    p_curr: float
        current price
    
    p_prec: float
        previous price
    
    """
    return (p_curr / p_prec) - 1


def get_log_return(p_curr, p_prec):
    """Given p_curr (P_t) and p_prec (P_{t-1}) returns the log return"""
    return math.log(p_curr / p_prec)


def pick_common_obs_interval(time_series_collection):
    """Given a dictionary of time series dataframes (not necessarily prices) related to multiple stocks, it returns dataframe observed in the common widest time horizon
    Dictionary keys are stock ticker symbols
    """
    merged_data = pd.DataFrame()
    for symbol in time_series_collection.keys():
        ts = pd.DataFrame(time_series_collection[symbol])
        ts.rename(columns={ts.columns.tolist()[0]: f"{symbol}"}, inplace=True)
        if merged_data.empty:
            merged_data = ts
        else:
            merged_data = pd.merge(
                merged_data,
                ts,
                how="inner",
                left_index=True,
                right_index=True,
            )
    return merged_data


def get_return_from_closing(day_hourly_prices, ticker_symbol, type="simple"):
    """Given a list of hourly prices of a Stock (sequentially ordered by the index number) related to single observation time-horizon (e.g. day), hourly returns are computed and returned as sequentially indexed dataframe"""
    n = len(day_hourly_prices)
    returns = [0] * (n - 1)
    if type == "simple":
        for i in range(n - 1):
            returns[i] = get_simple_return(
                day_hourly_prices[i + 1], day_hourly_prices[i]
            )
    else:
        for i in range(n - 1):
            returns[i] = get_log_return(day_hourly_prices[i + 1], day_hourly_prices[i])
    returns = pd.DataFrame(
        returns, columns=[f"{ticker_symbol}"], index=range(n - 1)
    )
    return returns


def get_daily_stocks_returns(
    symbols, start_date, end_date, interval="1h",
):
    """Given a list of ticker, returns a Dictionary: for each day  in the interval (from start_date to end_date) and for each stock symbol, retrieves the hourly closing prices data frame (indexed by date-time) and computes the daily return matrix
    In case data are not available in the specified interval of time, will return maximum amount of available data of such interval
    """
    time_series = {}
    i_date = start_date
    while i_date != get_next_date(end_date, 1):
        day = pd.DataFrame()
        next_date = get_next_date(i_date, 1)
        for stock in symbols:
            hf_day_prices = get_closing(stock, i_date, next_date, interval, dest_path=None)
            daily_returns = get_return_from_closing(hf_day_prices.to_list(), stock)
            if day.empty:
                day = daily_returns
            else:
                day = pd.merge(
                day,
                daily_returns,
                how="inner",
                left_index=True,
                right_index=True,
            )
        time_series[i_date] = day
        i_date = next_date
    return time_series




if __name__ == "__main__":
    my_stocks =["AAPL","MSFT","META"]
    history = get_daily_stocks_returns(my_stocks, "2024-05-20", "2024-05-23")
    dest_path = ""
    for day in history:
        daily_data = history[day].dropna()
        dest_path = os.path.join(
            dest_path,  "returns_" + day + ".csv"
        )
        daily_data.to_csv(dest_path, index=True)
        dest_path =""
    plot_histograms(history["2024-05-20"], 300, "returns")
    
