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
import torch.nn.functional as F
import math
from model import Dual_BDSTN

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

def split_data(data, adj, timestep, feature_size, output_size):
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

    # spilt dataset into three parts: training(0.7)、validating(0.15) and predicting(0.15)
    train_size = dataX.shape[0] * 0.7
    validation_size = dataX.shape[0] * 0.15
    prediction_size = dataX.shape[0] * 0.15

    x_train = dataX[: train_size, :].reshape(-1, timestep, feature_size)
    y_train = dataY[: train_size].reshape(-1, output_size)
    adj_train = adjlist[: train_size]

    x_test = dataX[train_size:, :].reshape(-1, timestep, feature_size)
    y_test = dataY[train_size:].reshape(-1, output_size)
    adj_test = adjlist[train_size:]

    return [x_train, y_train, x_test, y_test, adj_train, adj_test]

class Config():
    data_path = 'pro_data_c.csv'
    adj_path = 'pro_data_adj.csv'
    sequence_len = 10
    batch_size = 64
    spatial_feature_size = 2
    temporal_feature_size = 2
    hidden_size = 64
    output_size = 2
    machine = 13
    num_layers = 1
    epochs = 100
    nhead = 1
    dropout = 0.2
    best_loss = 0
    learning_rate = 0.001
    model_name = 'Dual_BDSTN'
    save_path = '{}.pth'.format(model_name)

config = Config()
setup_seed(2000)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
# dataset
df = pd.read_csv(config.data_path)
adj = pd.read_csv(config.adj_path)
df = np.array(df)
adj = np.array(adj).reshape(df.shape[0], -1, df.shape[1])
# normalize
scaler = MinMaxScaler(feature_range=(0, 1))
df = scaler.fit_transform(df)
# A+I for each adj
adj = preprocess_adj(adj)

#  split dataset
x_train, y_train , x_test, y_test, adj_train, adj_test = split_data(df, adj,config.timestep,
                                                                    config.feature_size,
                                                                    config.output_size)

# transform dataset into tensor
x_train_tensor = torch.from_numpy(x_train).to(torch.float32).to(device)
y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
adj_train_tensor = torch.from_numpy(adj_train).to(torch.float32).to(device)
x_test_tensor = torch.from_numpy(x_test).to(torch.float32).to(device)
y_test_tensor = torch.from_numpy(y_test).to(torch.float32)
adj_test_tensor = torch.from_numpy(adj_test).to(torch.float32).to(device)

train_data = TensorDataset(x_train_tensor, y_train_tensor, adj_train_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor, adj_test_tensor)

train_loader = torch.utils.data.DataLoader(train_data, config.batch_size, False)
test_loader = torch.utils.data.DataLoader(test_data, config.batch_size, False)

# define model、loss function、and optimizer
model = Dual_BDSTN(config.spatial_feature_size, config.machine * config.temporal_feature_size,
                   config.hidden_size, config.output_size * config.machine, config.machine,
                   config.sequence_len, config.dropout, config.num_layers, config.nhead)
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
        optimizer.zero_grad()
        y_train_pred = model(x_train, adj_train).to(torch.device("cpu"))
        loss = loss_function(y_train_pred, y_train)
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
            y_test_pred = model(x_test, adj_test).to(torch.device("cpu"))
            test_loss = loss_function(y_test_pred, y_test)
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
y_pred_pred = model(x_pred_tensor, adj_pred_tensor, device).to(torch.device("cpu"))
y_pred_pred = scaler.inverse_transform(y_pred_pred.detach().numpy())
y_pred_tensor = scaler.inverse_transform(y_pred_tensor.detach().numpy())
y_pred_pred[y_test_pred < 0] = 0
plt.figure(figsize=(16, 6))
plt.title(config.model_name)
# show the last 20 step
plt.plot(range(0, 20), y_test_pred[-20:, 2], 'r', label='a_pred', marker='.')
plt.plot(range(0, 20), y_test_tensor[-20:, 2], 'b', label='a_tst', marker='*')
plt.show()
