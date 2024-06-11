import torch
import numpy as np




def is_spd(matrix):
    """
    Checks if matrix is Positive Semidefinite (PSD) by using cholesky decomposition

    Args:
    matrix: (np.ndarray || torch.Tensor)
        Input matrix

    Returns:
    (bool): 
        True if matrix is PSD, False otherwise.
    """
    try:
        # Attempt Cholesky decomposition
        if isinstance(matrix, torch.Tensor):
            numpy_matrix = matrix.numpy()
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False
    

def get_block_diagonal_tensor(*args):
    """
    Builds a Block Diagonal Matrix as Torch Tensor, using input matrices

    Args:
    *args (2D list): 
        one or more matrices

    Returns:
    list: 
        The result is a 2D Torch Tensor.
    """
    i_tensors = []
    for arg in args:
        i_tensors.append(torch.from_numpy(np.array(arg)).clone())
    return torch.block_diag(*i_tensors).tolist()


def get_kronecker_product(matrix_a, matrix_b):
    """
    Computes the Kronecker product between input matrices

    Args:
    matrix_a (np.ndarray): first matrix
    matrix_b (np.ndarray): second matrix

    Returns:
    list: 
        result of kronecker product (still a 2D Torch Tensor).
    """
    # Convert inputs to NumPy arrays if they are PyTorch tensors
    tensor_a = torch.Tensor(matrix_a)
    tensor_b = torch.Tensor(matrix_b)
    return torch.kron(tensor_a, tensor_b).tolist()

def symmetric(A):
    return 0.5 * (A + A.t())


    
def orthogonal_projection(A, B):
    out = A - B.mm(symmetric(B.transpose(0,1).mm(A)))
    return out

def retraction(A, ref):
    data = A + ref
    Q, R = data.qr()
    # To avoid (any possible) negative values in the output matrix, we multiply the negative values by -1
    sign = (R.diag().sign() + 0.5).sign().diag()
    out = Q.mm(sign)
    return out



if __name__=="__main__":
    x = [[5,5],[9,9]],[[3,2],[1,1]]
    print(get_block_diagonal_tensor(x[0], x[1]))
    print(get_kronecker_product(x[0], x[1]))