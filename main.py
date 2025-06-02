import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset
from tqdm import tqdm
import random
import os
import time
from model import Dual_BDSTN
from sklearn import metrics

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def preprocess_adj(adj):
    adj_list = []
    for i in range(len(adj)):
        matrix_i = torch.eye(adj[i].shape[0])
        adj_i = matrix_i + adj[i]
        degree_matrix = torch.sum(adj_i, dim=1, keepdim=False)
        degree_matrix = degree_matrix.pow(-1)
        degree_matrix[degree_matrix == float("inf")] = 0.
        degree_matrix = torch.diag(degree_matrix)
        adj_list.append((torch.mm(degree_matrix, adj_i)).numpy())
    return np.array(adj_list)

def split_data(data, adj, timestep, feature_size):
    dataX = []
    dataY = []
    adjlist = []

    # one step ahead
    for index in range(len(data) - timestep):
        dataX.append(data[index: index + timestep])
        dataY.append(data[index + timestep])
        adjlist.append(adj[index: index + timestep])
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    adjlist = np.array(adjlist)

    # spilt dataset into three parts: training(0.7), validating(0.15) and predicting(0.15)
    train_size = int(dataX.shape[0] * 0.7)
    validation_size = int(dataX.shape[0] * 0.15)
    prediction_size = dataX.shape[0] - train_size - validation_size

    x_train = dataX[: train_size, :].reshape(-1, timestep, feature_size)
    y_train = dataY[: train_size].reshape(-1, feature_size)
    adj_train = adjlist[: train_size]
    x_test = dataX[train_size: (train_size + validation_size), :].reshape(-1, timestep, feature_size)
    y_test = dataY[train_size: (train_size + validation_size)].reshape(-1, feature_size)
    adj_test = adjlist[train_size:(train_size + validation_size)]
    x_pred = dataX[-prediction_size:, :].reshape(-1, timestep, feature_size)
    y_pred = dataY[-prediction_size:].reshape(-1, feature_size)
    adj_pred = adjlist[-prediction_size:]

    return [x_train, y_train, x_test, y_test, x_pred, y_pred, adj_train, adj_test, adj_pred]

class Config():
    data_path = 'data_all.csv'
    adj_path = 'data_adj.csv'
    timestep = 8
    batch_size = 32
    spatial_feature_size = 2
    temporal_feature_size = 2
    hidden_size = 32
    output_size = 4
    machine = 13
    num_layers = 2
    epochs = 100
    nhead = 8
    dropout = 0.2
    best_loss = 0  # If required, the value 9999 can be assigned as a default value.
    learning_rate = 0.0001
    model_name = 'Dual_BDSTN'
    save_path = '{}.pth'.format(model_name)

config = Config()
setup_seed(2000)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
# dataset
df = pd.read_csv(config.data_path)
adj = pd.read_csv(config.adj_path)
dfk = np.array(df)
adj = np.array(adj).reshape(df.shape[0], -1, config.machine)
# normalize
scaler = MinMaxScaler(feature_range=(0, 1))
df = scaler.fit_transform(df)
# split input features into spatial and temporal features
df_snum = []
df_tnum = []
for i in range(0, df.shape[1], config.temporal_feature_size + config.spatial_feature_size):
    df_snum.extend([i, i + config.spatial_feature_size - 1])
    df_tnum.extend([i + config.spatial_feature_size, i + config.temporal_feature_size + 1])
df_s = df[:, np.array(df_snum)]  # spatial features
df_t = df[:, np.array(df_tnum)]  # temporal features
# Fuse features to support downstream subsequent separation
df_st = np.concatenate((df_s, df_t), axis=1)
# A+I for each adj
adj = preprocess_adj(adj)

#  split dataset
x_train, y_train, x_test, y_test, x_pred, y_pred, adj_train, adj_test, adj_pred = split_data(df_st, adj, config.timestep,
                                                                                             (config.spatial_feature_size
                                                                                              + config.temporal_feature_size)
                                                                                             * config.machine)

# transform dataset into tensor
x_train_tensor = torch.from_numpy(x_train).to(torch.float32).to(device)
y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
adj_train_tensor = torch.from_numpy(adj_train).to(torch.float32).to(device)
x_test_tensor = torch.from_numpy(x_test).to(torch.float32).to(device)
y_test_tensor = torch.from_numpy(y_test).to(torch.float32)
adj_test_tensor = torch.from_numpy(adj_test).to(torch.float32).to(device)
x_pred_tensor = torch.from_numpy(x_pred).to(torch.float32).to(device)
y_pred_tensor = torch.from_numpy(y_pred).to(torch.float32)
adj_pred_tensor = torch.from_numpy(adj_pred).to(torch.float32).to(device)

train_data = TensorDataset(x_train_tensor, y_train_tensor, adj_train_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor, adj_test_tensor)
pred_data = TensorDataset(x_pred_tensor, y_pred_tensor, adj_pred_tensor)

train_loader = torch.utils.data.DataLoader(train_data, config.batch_size, False)
test_loader = torch.utils.data.DataLoader(test_data, config.batch_size, False)
pred_loader = torch.utils.data.DataLoader(pred_data, config.batch_size, False)

# define model、loss function、and optimizer
model = Dual_BDSTN(config.spatial_feature_size, config.machine * config.temporal_feature_size,
                   config.hidden_size, config.output_size * config.machine, config.machine,
                   config.timestep, config.dropout, config.num_layers, config.nhead).to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)

# train model
train_loss = []
vali_loss = []
train_cor = []
vali_cor = []
start_time = time.time()
for epoch in range(config.epochs):
    model.train()
    running_loss = 0
    testing_loss = 0
    train_bar = tqdm(train_loader)
    k = 0
    n = 0
    s = time.time()
    for data in train_bar:
        x_train, y_train, adj_train = data
        s_x_train = x_train[:, :, :config.machine * config.spatial_feature_size]
        t_x_train = x_train[:, :, config.machine * config.spatial_feature_size:]
        s_x_train = s_x_train.reshape(s_x_train.shape[0], s_x_train.shape[1], config.machine, -1)
        optimizer.zero_grad()
        y_train_pred = model(s_x_train, adj_train, t_x_train).to(torch.device("cpu"))
        loss = loss_function(y_train_pred.reshape(y_train_pred.shape[0], -1), y_train)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        n += 1
        train_bar.desc = "train epoch[{}/{}] loss:{:.5f}".format(epoch + 1, config.epochs, loss)
    train_loss.append(running_loss / n)
    e = time.time()
    print(e-s)

    # validate
    model.eval()
    test_loss = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for data in test_bar:
            x_test, y_test, adj_test = data
            s_x_test = x_test[:, :, :config.machine * config.spatial_feature_size]
            t_x_test = x_test[:, :, config.machine * config.spatial_feature_size:]
            s_x_test = s_x_test.reshape(s_x_test.shape[0], s_x_test.shape[1], config.machine, -1)
            y_test_pred = model(s_x_test, adj_test, t_x_test).to(torch.device("cpu"))
            test_loss = loss_function(y_test_pred.reshape(y_test_pred.shape[0], -1), y_test)
            k = k + 1
            testing_loss += test_loss.item()
            test_bar.desc = "test epoch[{}/{}] loss:{:.5f}".format(epoch + 1, config.epochs, test_loss)
        vali_loss.append(testing_loss / k)

    if test_loss < config.best_loss:
        config.best_loss = test_loss
        torch.save(model.state_dict(), config.save_path)
end_time = time.time()
training_duration = end_time - start_time
print(f"Time of Training and Validating: {training_duration} s")
print('Finished ')

# plot the training and validate loss
plt.title('loss', fontsize = 15)
plt.plot(range(0, config.epochs), train_loss, label='train')
plt.plot(range(0, config.epochs), vali_loss, label='valid')
plt.legend(fontsize="10", loc="upper right")
plt.show()

# prediction
s_x_pred_tensor = x_pred_tensor[:, :, :config.machine * config.spatial_feature_size]
t_x_pred_tensor = x_pred_tensor[:, :, config.machine * config.spatial_feature_size:]
s_x_pred_tensor = s_x_pred_tensor.reshape(s_x_pred_tensor.shape[0], s_x_pred_tensor.shape[1], config.machine, -1)
y_pred_pred = model(s_x_pred_tensor, adj_pred_tensor, t_x_pred_tensor).to(torch.device("cpu"))


y_pred_pred = (y_pred_pred.detach().numpy()).reshape(y_pred_pred.shape[0], -1)
y_pred_tensor = (y_pred_tensor.detach().numpy()).reshape(y_pred_tensor.shape[0], -1)
y_pred_pred = scaler.inverse_transform(y_pred_pred)
y_pred_tensor = scaler.inverse_transform(y_pred_tensor)
y_pred_pred[y_pred_pred < 0] = 0
# show the last 20 step (your selected labels) and
# column 0- config.machine * config.feature_spatial_size is the predicted indicator
# plt.figure(figsize=(16, 6))
plt.title(config.model_name)
plt.plot(range(0, y_pred_pred.shape[0]), y_pred_pred[:, 8], 'r', label='a_pred', marker='.')
plt.plot(range(0, y_pred_tensor.shape[0]), y_pred_tensor[:, 8], 'b', label='a_tst', marker='*')
plt.show()
plt.title(config.model_name)
plt.plot(range(0, y_pred_pred.shape[0]), y_pred_pred[:, 9], 'r', label='a_pred', marker='.')
plt.plot(range(0, y_pred_tensor.shape[0]), y_pred_tensor[:, 9], 'b', label='a_tst', marker='*')
plt.show()
plt.title(config.model_name)
plt.plot(range(0, y_pred_pred.shape[0]), y_pred_pred[:, 18], 'r', label='a_pred', marker='.')
plt.plot(range(0, y_pred_tensor.shape[0]), y_pred_tensor[:, 18], 'b', label='a_tst', marker='*')
plt.show()
plt.title(config.model_name)
plt.plot(range(0, y_pred_pred.shape[0]), y_pred_pred[:, 19], 'r', label='a_pred', marker='.')
plt.plot(range(0, y_pred_tensor.shape[0]), y_pred_tensor[:, 19], 'b', label='a_tst', marker='*')
plt.show()

# MAE and RMSE for indicators of all machines in a production line.
print(metrics.mean_absolute_error(y_pred_pred[:, :config.machine * config.spatial_feature_size],
                                  y_pred_tensor[:, :config.machine * config.spatial_feature_size]))
print(np.sqrt(metrics.mean_squared_error(y_pred_pred[:, :config.machine * config.spatial_feature_size],
                                         y_pred_tensor[:, :config.machine * config.spatial_feature_size])))