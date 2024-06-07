import torch
from torch import nn

#from numpy.linalg import matrix_rank

from spdnet.data.linalg import is_spd
from spdnet.data.utils import make_spd_matrix


class BiMap(nn.Module):
    """
    Equivalent to BiMap Layer of some papers on SPDNET
    """

    def __init__(self, in_size, out_size):
        """
        Args:
        in_size: int
            size of the single dimension of input (square) matrix
            (es: for a NxN matrix, in_size = N)
        out_size: int
            size of the single output (square) matrix
            (es: for a NxN matrix, out_size = N)

        Returns:
        calls constructor, and initializes weights
        """
        super(BiMap, self).__init__()
        self.W = nn.Parameter(torch.rand((out_size, in_size)).normal_())  #weights initialization following stand. normal distrib.


    def forward(self, X):
        """
        Args:
        X: torch.Tensor
            Is the input matrix of size (in_size x in_size)

        Returns:
        The output of forward step, that is W^T*X*W  
        (matrix multiplication of input matrix with weights)
        
        """       
        x = X.squeeze(0)
        w_t = self.get_weights()
        res = torch.mm(w_t,x)
        res = torch.mm(res, torch.transpose(w_t,0,1))
        return res
    
    def get_weights(self):
        """
        Returns:
        the weights of this layer
        """
        return self.W

    """  def check_rank(self):
        # sanity check, if weights matrix is degenerate
        return min(self.W.data.size()) == matrix_rank(self.W.data.numpy()) """
    


class ReEig(nn.Module):
    """
    Also known as ReEig layer (in some papers on SPDNets)
    Enables non-linearity by applying ReLU block for every element of matrix.


    Returns:
    torch.autograd.Variable object 
        returns SPD matrix of the same size, result of nullification of negative elements of input matrix.
    """
    def __init__(self):
        super(ReEig, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, X):
        return self.relu(X)



class RiemSPD(nn.Module):
    """
    Class used to instantiate the network,
    Network will take an SPD input matrix of size (NxN),
    will return an SPD output matrix of same size (NxN)
    by applying reductions in several layers, while preserving SPD properties.


    Args:
    in_size: int
        size of the single dimension of input (square) matrix
        (es: for a NxN matrix, in_size = N)
    
    """

    def __init__(self, in_size):
        super(RiemSPD, self).__init__()
        self.n_features = in_size
        
        self.encoder = nn.Sequential(
            BiMap(in_size, 5),
            ReEig(),
            nn.Tanh()
        )
        
        self.decoder = nn.Sequential(
            BiMap(5, in_size),
            nn.ReLU()
        )
        
    def forward(self, X):
        return self.decoder(self.encoder(X))





if __name__ == "__main__":
    bimap_l = BiMap(2,3)
    regeig_l = ReEig()
    x = make_spd_matrix(2)
    x = torch.Tensor(x).unsqueeze(0)
    x1 = bimap_l.forward(x)
    print("weight:\n", bimap_l.get_weights())
    print("result:\n",x1)
    x2= regeig_l(x1)
    print("reg \n:",x2)
    #check if they are spd
    print("is_x_spd?:", is_spd(x1.detach().numpy()))
    print("is_x1_spd?:", is_spd(x1.detach().numpy()))
    print("is_x1_spd?:", is_spd(x2.detach().numpy()))
    net = RiemSPD(2)
    print(net(x))




    


