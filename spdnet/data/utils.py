from sklearn.datasets import make_spd_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import numpy as np



def make_financial_spd_matrix(n_stocks, n_obs):
    """
    Creates a time series of simulated RCOV matrices, correlated in time
    PROBLEM: Does not guarantee SPD matrices so we use numpy spd matrix generator instead

    Args:
    n_stocks: int
        it's the size of the matrix (NxN where N=n_stocks)
    
    n_obs: int
        it's the number of matrices in the list, to be generated

    Returns:
    list of matrices (2D list)
    """
    Sigma_zero =  make_spd_matrix(n_stocks).tolist()
    time_series = [Sigma_zero]
    for t in range(1,2*n_obs):
        #discard the first half, they'll be less likely correlated in time.
        Sigma_err = make_spd_matrix(n_stocks).tolist()
        Sigma_t = np.array(time_series[t-1])*0.5 + Sigma_err
        time_series.append(Sigma_t.tolist())
    return time_series[-n_obs:][0]





def get_next_date(date_string, x):
    """Given a YY-MM-DD date string, it returns the date string of next (x) days"""
    current_date = datetime.strptime(date_string, "%Y-%m-%d")
    next_date = current_date + timedelta(days=x)
    return next_date.strftime("%Y-%m-%d")


def plot_histograms(data, bins, xLabel):
    """Assuming to have a dataframe as input
    """
    n_graphs = data.shape[1]
    plt.figure(figsize=(1, n_graphs))

    for stock in data.columns:
        plt.hist(data[stock], bins=bins, alpha=0.4, label=stock)
    plt.xlabel(xLabel)
    plt.ylabel("frequency")
    plt.legend(loc="upper right")
    plt.show()


def plot_prices(data):
    """Assuming to have a dataframe as input
    """
    n_graphs = data.shape[1]
    plt.figure(figsize=(1, n_graphs))
    for stock in data.columns:
        plt.plot(data[stock], label=f'{stock}')

    plt.title('Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()



if __name__=="__main__":
    matrix = make_spd_matrix(2)  #generates a SPD matrix of size NXN (N=2)
    print(matrix)

    print(make_financial_spd_matrix(2, 1))

