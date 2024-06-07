import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.modules.loss import MSELoss
from torch import FloatTensor
from torch import optim
from spdnet.net.SPDnet import RiemSPD
from spdnet.net.dataset import RCOVDataset, extract_x_y_from_tseries
from spdnet.data.utils import make_spd_matrix



#ARGS
N_OBS=1000
N_STOCK = 2
N_LAGS = 1


train_data = [make_spd_matrix(N_STOCK).tolist() for _ in range(N_OBS)]
print(train_data)
dict = extract_x_y_from_tseries(train_data,N_LAGS,'d')
train_dataset = RCOVDataset(dict['x'], dict['y'])

print("dataset.x ", train_dataset.x, "\n")
print("dataset.y ", train_dataset.y, "\n")

network = RiemSPD(N_STOCK)
optimizer = optim.Adam(network.parameters(), lr=0.1)
criterion = MSELoss()
loss_hist = []



for x, y  in train_dataset:
        x_i = torch.Tensor(x)
        y_i = torch.Tensor(y)

        print('x_i',x_i)

        optimizer.zero_grad()
        prediction = network(x_i)
        loss = criterion(prediction, y_i)
        loss_i = loss.item()
        loss.backward()

        print("TRAIN LOSS: {0}".format(loss_i))

        loss_hist.append((loss_i))
        optimizer.step()

plt.plot(loss_hist, label='loss')
plt.legend()
plt.show()





