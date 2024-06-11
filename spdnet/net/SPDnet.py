import torch
from torch import nn
import numpy as np
from spdnet.data.linalg import is_spd
from spdnet.data.utils import make_spd_matrix


class StiefelParameter(nn.Parameter):
    """
        Creates parameters that are constrained to lie on the Stiefel manifold. 
        Rmk: Stiefel manifold is the set of all orthogonal matrices of a given dimension
    """
    def __new__(cls, data=None, requires_grad=True):
        #returns an instance of the class, calling the parent class (nn.Parameter)
        return super(StiefelParameter, cls).__new__(cls, data, requires_grad=requires_grad)

    def __repr__(self):
        return 'Parameter:' + self.data.__repr__()


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
        calls constructor, and initializes weights as orthogonal (semi-orthogonal matrix)
        """
        super(BiMap, self).__init__()
        self.W = StiefelParameter(torch.FloatTensor(in_size, out_size), requires_grad=True)
        nn.init.orthogonal_(self.W)



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
        res = x
        weight = self.W
        res = torch.mm(res, weight)
        res = torch.mm(weight.T, res)
        return res
    
    def get_weights(self):
        """
        Returns:
        the weights of this layer
        """
        return self.W

class Rectify(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, epsilon):
        """
        The method computes the SVD of the input matrix and then rectifies its singular values to be above the specified threshold (epsilon)
        """

        ctx.save_for_backward(X, epsilon)
        u, s, v = X.svd()
        s[s<epsilon] = epsilon
        res = u.mm(s.diag().mm(u.t()))
        return res


    @staticmethod
    def backward(ctx, grad_output):
        """
        The backward method computes the gradients with respect to the input matrix using the chain rule and gradients of the SVD operation.
        """

        X, epsilon = ctx.saved_variables
        grad_input = None

        if ctx.needs_input_grad[0]:
            u, s, v = X.svd()

            max_mask = s > epsilon
            s_max = torch.where(max_mask, s, epsilon)
            dLdV = 2 * (grad_output.mm(u) * s_max.unsqueeze(0))
            dLdS = torch.eye(s.size(0), device=s.device) * (max_mask.float().matmul(grad_output).matmul(u.t()))
            grad_input = u @ (dLdV - dLdV.t() + dLdS)

            
        return grad_input, None


class ReEig(nn.Module):
    """
    Also known as ReEig layer (in some papers on SPDNets)
    Enables non-linearity by applying Rectification block  ensuring that eigenvalues of the input matrix are above a certain threshold (determined by epsilon)
    We follow the implementation described in 'Huang, Z., & Van Gool, L. J. (2017, February). A Riemannian Network for SPD Matrix Learning'

    Returns:
    torch.autograd.Variable object 
        returns SPD matrix of the same size, result of nullification of negative elements of input matrix.
    """
    def __init__(self, epsilon = 1e-4):
        super(ReEig, self).__init__()
        self.register_buffer('epsilon', torch.FloatTensor([epsilon])) #creates a persistent tensor
    

    def forward(self, X):
        return Rectify.apply(X, self.epsilon)



class RiemSPD(nn.Module):
    """
    Class used to instantiate the network,
    Network will take an SPD input matrix of size (MxM),
    will return an SPD output matrix of size (NxN)
    by applying reductions in several layers, while preserving SPD properties.


    Args:
    in_size: int
        size of the single dimension of input (square) matrix
        (es: for a MxM matrix, in_size = M)
    out_size: int 
        size of the single dimension of output (square) matrix
        (es: for a NxN matrix, out_size = N)
    
    """

    def __init__(self, in_size, out_size):
        super(RiemSPD, self).__init__()
        
        self.encoder = nn.Sequential(
            BiMap(in_size, 5),
            ReEig(),
            #nn.Tanh()
        )
        
        self.decoder = nn.Sequential(
            BiMap(5, out_size),
            ReEig()
        )
        
    def forward(self, X):
        return self.decoder(self.encoder(X))





if __name__ == "__main__":
    bimap_l = BiMap(2,6)
    regeig_l = ReEig()
    x = make_spd_matrix(2)
    x = torch.Tensor(x).unsqueeze(0)
    print("x",x)
    x1 = bimap_l.forward(x)
    print("weight:\n", bimap_l.get_weights())
    print("result:\n",x1)
    x2= regeig_l(x1)
    print("reg \n:",x2)
    #check if they are spd
    print("is_x_spd?:", is_spd(x1.detach().numpy()))
    print("is_x1_spd?:", is_spd(x1.detach().numpy()))
    print("is_x1_spd?:", is_spd(x2.detach().numpy()))
    net = RiemSPD(2,3)
    print(net(x))




    


