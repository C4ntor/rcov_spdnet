import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.modules.loss import MSELoss
from torch import FloatTensor
from torch import optim
from spdnet.net.SPDnet import RiemSPD
from spdnet.net.dataset import RCOVDataset, extract_x_y_from_tseries
from spdnet.data.utils import make_spd_matrix
from spdnet.data.linalg import is_spd



#ARGS
N_OBS=1000
N_STOCK = 2
N_LAGS = 20

train_data = [make_spd_matrix(N_STOCK).tolist() for _ in range(N_OBS)]
dict = extract_x_y_from_tseries(train_data,N_LAGS,'d')

#PLEASE NOTICE THAT KRONECKER PRODUCT MODE IS EXPERIMENTAL!

train_dataset = RCOVDataset(dict['x'], dict['y'])



M_SIZE=len(train_dataset.x[0])



print("dataset.x ", train_dataset.x, "\n")

print("dataset.y ", train_dataset.y, "\n")


network = RiemSPD(M_SIZE,N_STOCK)
optimizer = optim.Adam(network.parameters(), lr=0.1)
criterion = MSELoss()
loss_hist = []

h_real_var_01 = []              #
h_pred_var_01 = []              #

for x, y  in train_dataset:
        x_i = torch.Tensor(x)
        assert(is_spd(x_i.detach().numpy()))
        y_i = torch.Tensor(y)
        assert(is_spd(y_i.detach().numpy()))

        h_real_var_01.append(y_i[0][1].item())         #

        optimizer.zero_grad()
        prediction = network(x_i)
        assert(is_spd(prediction.detach().numpy()))
        h_pred_var_01.append(prediction[0][1].item())  #

        print('x_i',x_i)
        print('y_i',y_i)
        print('y_i_PRED',prediction)

        loss = criterion(prediction, y_i)
        loss_i = loss.item()
        loss.backward()

        print("TRAIN LOSS: {0}".format(loss_i))

        loss_hist.append((loss_i))
        optimizer.step()

#plt.plot(loss_hist, label='loss')

plt.plot(h_pred_var_01, 'b', label='predicted') #
plt.plot(h_real_var_01, 'r', label='real') #


plt.legend()
plt.show()





