from sklearn.datasets import make_spd_matrix
import numpy as np


def make_financial_spd_matrix(n_stocks, n_obs):
    """
    Creates a time series of simulated RCOV matrices, correlated in time

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




if __name__=="__main__":
    matrix = make_spd_matrix(2)  #generates a SPD matrix of size NXN (N=2)
    print(matrix)

    print(make_financial_spd_matrix(2, 1))

