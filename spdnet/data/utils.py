from sklearn.datasets import make_spd_matrix
import numpy as np
import torch





if __name__=="__main__":
    matrix = make_spd_matrix(2)  #generates a SPD matrix of size NXN (N=2)
    print(matrix)

