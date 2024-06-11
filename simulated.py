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
from spdnet.net.optimizer import StiefelMetaOptimizer


#ARGS
N_OBS=1000
N_STOCK = 2
N_LAGS = 4
SEED = 1

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


train_data = [make_spd_matrix(N_STOCK).tolist() for _ in range(N_OBS)]
#creates a list (time series) of SPD matrices with size (N_STOCK X N_STOCK).
#the list contains (N_OBS) matrices


dict = extract_x_y_from_tseries(train_data,N_LAGS,'d')
#preprocess the generated matrices, using BLOCK DIAGONAL MODE, preparing them to be included in RCOVDataset (as x, and y)


#PLEASE NOTICE THAT KRONECKER PRODUCT MODE ('k') IS STILL EXPERIMENTAL!
train_dataset = RCOVDataset(dict['x'], dict['y'])



M_SIZE=len(train_dataset.x[0])
#determines the size of the diagonal block matrix or krockere matrix, based on the lags and input size


print("dataset.x ", train_dataset.x, "\n")

print("dataset.y ", train_dataset.y, "\n")


network = RiemSPD(M_SIZE,N_STOCK)
optimizer = torch.optim.SGD(network.parameters(), lr=0.05)
optimizer = StiefelMetaOptimizer(optimizer)
criterion = MSELoss()
loss_hist = []
i=0
baseline_pred = None
time_step = N_LAGS-1
cumulative_loss=0
mse_basel= 0
loss_baseline = []
var_00_pred = []
var_00_true = []

for x, y  in train_dataset:
        x_i = torch.Tensor(x)
        assert(is_spd(x_i.detach().numpy()))
        y_i = torch.Tensor(y)
        assert(is_spd(y_i.detach().numpy()))
        baseline_pred = torch.Tensor(train_data[time_step])
        var_00_true.append(y_i[0][0].detach().numpy())
        optimizer.zero_grad()
        prediction = network(x_i)
        assert(is_spd(prediction.detach().numpy()))
        var_00_pred.append(prediction[0][0].detach().numpy())
        print('x_i',x_i)
        print('y_i',y_i)
        print('y_i_basel_PRED', baseline_pred)
        print('y_i_PRED',prediction)

        mse_basel = torch.mean((baseline_pred - y_i) ** 2)
        loss_baseline.append(mse_basel)
        loss = criterion(prediction, y_i)
        loss_i = loss.item()
        loss.backward()

        print("LOSS: {0}".format(loss_i))

        
        time_step += 1
        loss_hist.append((loss_i))
        optimizer.step()

torch.save(network.state_dict(), 'model.pth')


plt.plot(loss_hist, label='loss_network')
plt.plot(loss_baseline, label = 'loss_baseline')
plt.legend()
plt.show()

#we can load it as follows: loaded_model = RiemSPD(M_SIZE,N_STOCK
#loaded_model.load_state_dict(torch.load('model.pth'))
plt.plot(var_00_true, label='var_00 true')
plt.plot(var_00_pred, label = 'var_00 predicted')
plt.legend()
plt.show()
print("CUMULATIVE_LOSS_NET:", sum(loss_hist))
print("AVG_LOSS_NET:", sum(loss_hist)/N_OBS)
print("CUMULATIVE_LOSS_BASELINE:", sum(loss_baseline))
print("AVG_LOSS_BASELINE:", sum(loss_baseline)/N_OBS)






