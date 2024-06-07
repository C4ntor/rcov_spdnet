from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from spdnet.data.utils import make_spd_matrix
from spdnet.data.linalg import get_block_diagonal_tensor, get_kronecker_product


def extract_x_y_from_tseries(rcov_ts,n,mode):
    """
        Prepares the dataset, creating X and Y elements, starting from time series of Realized Covariance (RCOV) Matrices.
        Each X_i element will be a matrix, which encompasses information of (k) lagged RCOV matrices.
        X_i is build using Kronecker product or Block Diagonal matrix of the (k) lagged RCOV matrices, preceding the given time step (i): Sigma_{i-k}, ..., Sigma_{i-1} (included)
        Y_i will be the corresponding Sigma_i


        Args:
        rcov_ts (list): 
            a time series of matrices (all of same size) [Sigma_0, ....., Sigma_T] where the first element in the list, is the oldest obs, and Sigma_T (last element) is the most recent one
        n (int): 
            s the number of lags (in our specific case, the number of matrices) used as input to the model, in order to forecast the rcov matrix of the following time step
        mode (string): 
            can be 'k' for kronecker product or 'd' for diagonal block matrix

        Labels (y to be predicted) are determined as the first (matrix) occurence in the given time series that follows the latest (most recent) lag
        returns a dictionary in the form {'x':X_data, 'y':Y_data, 'b':b_data}
        Where b_data collects the lagged covariance matrix previous, to the predicted one, for each iteration
    """
     
    if mode not in ["k", "d"]:
        raise Exception("mode must be equal to 'k' or 'd'")

    if n >= len(rcov_ts):
        raise Exception("the number of lags must be between [1 and len(rcov_ts)-1]")

    x = []
    y = []
    b = []
    for i in range(n, len(rcov_ts)):
        obs = np.array(rcov_ts[i - n : i])
        if mode == "k":
            if len(obs) == 1:
                # if only one lag is used, do kronecker product of that obs with itself
                x.append(get_kronecker_product(obs[0], obs[0]))
            else:
                kron_res = get_kronecker_product(obs[0], obs[1])
                if len(obs)<=2:
                    x.append(kron_res)
                else:
                    for i in range(2, len(obs)):
                        kron_res = get_kronecker_product(kron_res, obs[i])
                        x.append((kron_res))
                

        if mode == "d":
            x.append(get_block_diagonal_tensor(*obs))
        b.append(rcov_ts[i-1])
        y.append(rcov_ts[i])
    
    return {'x':x, 'y':y, 'b':b}
    



class RCOVDataset(Dataset):
    def __init__(self, x, y):

        """
        Args:
            x_data (numpy array or torch tensor): Data for input features.
            y_data (numpy array or torch tensor): Data for output labels.
        """
     
        super(RCOVDataset, self).__init__()
        self.x = x
        self.y = y
        

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x =  self.x[idx]
        y = self.y[idx]
        return x,y


if __name__ == "__main__":
    matrix_size = 2
    dataset_size = 2
    n_lags = 1
    train_data = [make_spd_matrix(matrix_size).tolist() for _ in range(dataset_size)]
    print(train_data)
    
    dict = extract_x_y_from_tseries(train_data,n_lags,'d')
    

    train_dataset = RCOVDataset(dict['x'], dict['y'])
    print("dataset.x ", train_dataset.x, "\n")
    print("dataset.y ", train_dataset.y, "\n")


    for x, y  in train_dataset:
        print(torch.tensor(x))
        print("\n")
        print(y)
        print("====")

    print("*****")


    

    