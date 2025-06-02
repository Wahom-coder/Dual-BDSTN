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

    # Note that the separation is just for explanatory clarity and does not correspond to the true experiment.
    train_size = dataX.shape[0]
    x_train = dataX[: train_size, :].reshape(-1, timestep, feature_size)
    y_train = dataY[: train_size].reshape(-1, feature_size)
    adj_train = adjlist[: train_size]

    return [x_train, y_train, adj_train]

class Config():
    data_path = 'input.csv'
    adj_path = 'input_adj.csv'
    timestep = 4
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
    best_loss = 0  # If required, the value 9999 can be assigned as a default value.(not needed here in only training)
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
# Fuse spatial and temporal features to support downstream subsequent separation conveniently
df_st = np.concatenate((df_s, df_t), axis=1)
# A+I for each adj
adj = preprocess_adj(adj)

#  split dataset
x_train, y_train,  adj_train = split_data(df_st, adj, config.timestep, (config.spatial_feature_size
                                                                        + config.temporal_feature_size) * config.machine)
# transform dataset into tensor
x_train_tensor = torch.from_numpy(x_train).to(torch.float32).to(device)
y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
adj_train_tensor = torch.from_numpy(adj_train).to(torch.float32).to(device)
train_data = TensorDataset(x_train_tensor, y_train_tensor, adj_train_tensor)
train_loader = torch.utils.data.DataLoader(train_data, config.batch_size, False)

# define model, loss function, and optimizer
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
end_time = time.time()
training_duration = end_time - start_time
print(f"Time of Training and Validating: {training_duration} s")
print('Finished ')

print("Note that the subdataset are just for explanatory clarity.")
print("Owing to industrial confidentiality, only part of the data is used to demonstrate model training and execution.")