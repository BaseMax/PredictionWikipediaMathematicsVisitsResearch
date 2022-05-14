import time
import torch
from tqdm import tqdm
from numba import cuda
import tensorflow as tf
import torch.nn.functional
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU
from torch_geometric_temporal.dataset import WikiMathsDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

# SET NUMBA_ENABLE_CUDASIM=1
print(cuda.gpus)
# device_id = 0
# cuda.select_device(device_id)

print(torch.cuda.is_available())

print(tf.test.gpu_device_name())

print("__version__:", torch.__version__)

loader = WikiMathsDatasetLoader()


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, filters, k, linear_digit):
        super(RecurrentGCN, self).__init__()
        self.recurrent = GConvGRU(node_features, filters, k)
        # 100%|██████████| 50/50 [21:05<00:00, 25.30s/it]
        # MSE: 0.7935
        # now: 100%|██████████| 50/50 [20:35<00:00, 24.72s/it]
        # MSE: 0.7763
        # 3m 4s
        self.linear = torch.nn.Linear(filters, linear_digit)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
                
        h = self.linear(h)
        return h

tests = [
    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":14*4,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14*4,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":14*5,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14*5,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":14*6,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14*6,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":14*7,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14*7,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":14*3,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14*3,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":14*2,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14*2,
    #     "train_ratio":0.5
    # },
    {
        "k":2,
        "linear_digit":1,
        "node_features":14*4,
        "filters":32,
        "lr":0.01,
        "lags":14*4,
        "train_ratio":0.5
    },
    {
        "k":3,
        "linear_digit":1,
        "node_features":14*4,
        "filters":32,
        "lr":0.01,
        "lags":14*4,
        "train_ratio":0.5
    },
    {
        "k":3,
        "linear_digit":1,
        "node_features":14*5,
        "filters":32,
        "lr":0.01,
        "lags":14*5,
        "train_ratio":0.5
    },
    {
        "k":3,
        "linear_digit":1,
        "node_features":14*6,
        "filters":32,
        "lr":0.01,
        "lags":14*6,
        "train_ratio":0.5
    },
    {
        "k":3,
        "linear_digit":1,
        "node_features":14*7,
        "filters":32,
        "lr":0.01,
        "lags":14*7,
        "train_ratio":0.5
    },
    {
        "k":3,
        "linear_digit":1,
        "node_features":14*3,
        "filters":32,
        "lr":0.01,
        "lags":14,
        "train_ratio":0.5
    },
    {
        "k":3,
        "linear_digit":1,
        "node_features":14*2,
        "filters":32,
        "lr":0.01,
        "lags":14,
        "train_ratio":0.5
    },
    {
        "k":3,
        "linear_digit":1,
        "node_features":14*4,
        "filters":32,
        "lr":0.01,
        "lags":14*4,
        "train_ratio":0.5
    },
    {
        "k":4,
        "linear_digit":1,
        "node_features":14*4,
        "filters":32,
        "lr":0.01,
        "lags":14*4,
        "train_ratio":0.5
    },
    {
        "k":4,
        "linear_digit":1,
        "node_features":14*5,
        "filters":32,
        "lr":0.01,
        "lags":14*5,
        "train_ratio":0.5
    },
    {
        "k":4,
        "linear_digit":1,
        "node_features":14*6,
        "filters":32,
        "lr":0.01,
        "lags":14*6,
        "train_ratio":0.5
    },
    {
        "k":4,
        "linear_digit":1,
        "node_features":14*7,
        "filters":32,
        "lr":0.01,
        "lags":14*7,
        "train_ratio":0.5
    },
    {
        "k":4,
        "linear_digit":1,
        "node_features":14*3,
        "filters":32,
        "lr":0.01,
        "lags":14,
        "train_ratio":0.5
    },
    {
        "k":4,
        "linear_digit":1,
        "node_features":14*2,
        "filters":32,
        "lr":0.01,
        "lags":14,
        "train_ratio":0.5
    },
    {
        "k":4,
        "linear_digit":1,
        "node_features":14*4,
        "filters":32,
        "lr":0.01,
        "lags":14*4,
        "train_ratio":0.5
    },
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":14*4,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14*4,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":14*5,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14*5,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":14*6,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14*6,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":14*7,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14*7,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":14*3,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":14*2,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":3,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":4,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":5,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":10,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":50,
    #     "filters":20,
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":50,
    #     "filters":100,
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":80,
    #     "filters":100,
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":80,
    #     "filters":40,
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.02,
    #     "lags":14,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":3,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.02,
    #     "lags":14,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":4,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.02,
    #     "lags":14,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":5,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.02,
    #     "lags":14,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":10,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.02,
    #     "lags":14,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":80,
    #     "filters":100,
    #     "lr":0.02,
    #     "lags":14,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":80,
    #     "filters":40,
    #     "lr":0.02,
    #     "lags":14,
    #     "train_ratio":0.5
    # },


    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":50,
    #     "lr":0.02,
    #     "lags":14,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":16,
    #     "train_ratio":0.7
    # },
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":32,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":32,
    #     "train_ratio":0.7
    # },
    {
        "k":1,
        "linear_digit":1,
        "node_features":16,
        "filters":16,
        "lr":0.01,
        "lags":16,
        "train_ratio":0.7
    },
    {
        "k":1,
        "linear_digit":1,
        "node_features":64,
        "filters":16,
        "lr":0.01,
        "lags":64,
        "train_ratio":0.7
    },
    {
        "k":1,
        "linear_digit":1,
        "node_features":128,
        "filters":16,
        "lr":0.01,
        "lags":128,
        "train_ratio":0.7
    },
    {
        "k":1,
        "linear_digit":1,
        "node_features":256,
        "filters":16,
        "lr":0.01,
        "lags":256,
        "train_ratio":0.7
    },
    {
        "k":1,
        "linear_digit":1,
        "node_features":32,
        "filters":16,
        "lr":0.01,
        "lags":32,
        "train_ratio":0.7
    },
    {
        "k":2,
        "linear_digit":1,
        "node_features":32,
        "filters":16,
        "lr":0.01,
        "lags":32,
        "train_ratio":0.7
    },
    {
        "k":3,
        "linear_digit":1,
        "node_features":32,
        "filters":16,
        "lr":0.01,
        "lags":32,
        "train_ratio":0.7
    },
    {
        "k":1,
        "linear_digit":1,
        "node_features":2,
        "filters":16,
        "lr":0.01,
        "lags":2,
        "train_ratio":0.7
    },
    {
        "k":1,
        "linear_digit":1,
        "node_features":4,
        "filters":16,
        "lr":0.01,
        "lags":4,
        "train_ratio":0.7
    },
    {
        "k":1,
        "linear_digit":1,
        "node_features":8,
        "filters":16,
        "lr":0.01,
        "lags":8,
        "train_ratio":0.7
    },
    {
        "k":1,
        "linear_digit":1,
        "node_features":16,
        "filters":16,
        "lr":0.01,
        "lags":16,
        "train_ratio":0.7
    },
    {
        "k":1,
        "linear_digit":1,
        "node_features":16,
        "filters":2,
        "lr":0.01,
        "lags":16,
        "train_ratio":0.7
    },
    {
        "k":1,
        "linear_digit":1,
        "node_features":16,
        "filters":4,
        "lr":0.01,
        "lags":16,
        "train_ratio":0.7
    },
    {
        "k":1,
        "linear_digit":1,
        "node_features":16,
        "filters":8,
        "lr":0.01,
        "lags":16,
        "train_ratio":0.7
    },
    {
        "k":1,
        "linear_digit":1,
        "node_features":16,
        "filters":16,
        "lr":0.01,
        "lags":16,
        "train_ratio":0.7
    },
    {
        "k":1,
        "linear_digit":1,
        "node_features":16,
        "filters":32,
        "lr":0.01,
        "lags":16,
        "train_ratio":0.7
    },
    {
        "k":1,
        "linear_digit":1,
        "node_features":16,
        "filters":32,
        "lr":0.01,
        "lags":16,
        "train_ratio":0.7
    },
    {
        "k":1,
        "linear_digit":1,
        "node_features":16,
        "filters":32,
        "lr":0.02,
        "lags":16,
        "train_ratio":0.7
    },
    {
        "k":1,
        "linear_digit":1,
        "node_features":16,
        "filters":32,
        "lr":0.03,
        "lags":16,
        "train_ratio":0.7
    },
    {
        "k":1,
        "linear_digit":1,
        "node_features":16,
        "filters":16,
        "lr":0.01,
        "lags":16,
        "train_ratio":0.7
    },
    {
        "k":1,
        "linear_digit":2,
        "node_features":16,
        "filters":16,
        "lr":0.01,
        "lags":16,
        "train_ratio":0.7
    },
    {
        "k":1,
        "linear_digit":3,
        "node_features":16,
        "filters":16,
        "lr":0.01,
        "lags":16,
        "train_ratio":0.7
    },

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.3
    # },
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.4
    # },
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.6
    # },
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.7
    # },
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.9
    # },


    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":16,
    #     "train_ratio":0.3
    # },
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":16,
    #     "train_ratio":0.4
    # },
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":16,
    #     "train_ratio":0.5
    # },
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":16,
    #     "train_ratio":0.6
    # },
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":16,
    #     "train_ratio":0.7
    # },
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":16,
    #     "train_ratio":0.8
    # },
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":16,
    #     "train_ratio":0.9
    # },


    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":64,
    #     "lr":0.01,
    #     "lags":16,
    #     "train_ratio":0.7
    # },
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":64,
    #     "filters":2,
    #     "lr":0.01,
    #     "lags":64,
    #     "train_ratio":0.7
    # },
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":64,
    #     "filters":4,
    #     "lr":0.01,
    #     "lags":64,
    #     "train_ratio":0.7
    # },
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":64,
    #     "filters":8,
    #     "lr":0.01,
    #     "lags":64,
    #     "train_ratio":0.7
    # },
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":64,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":64,
    #     "train_ratio":0.7
    # },
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":64,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":64,
    #     "train_ratio":0.7
    # },
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":64,
    #     "filters":64,
    #     "lr":0.01,
    #     "lags":64,
    #     "train_ratio":0.7
    # },    
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":64,
    #     "filters":128,
    #     "lr":0.01,
    #     "lags":64,
    #     "train_ratio":0.7
    # },
    # {
    #     "k":3,
    #     "linear_digit":1,
    #     "node_features":256,
    #     "filters":2,
    #     "lr":0.01,
    #     "lags":256,
    #     "train_ratio":0.9
    # },
    # {
    #     "k":3,
    #     "linear_digit":1,
    #     "node_features":256,
    #     "filters":2,
    #     "lr":0.005,
    #     "lags":256,
    #     "train_ratio":0.9
    # },
    # {
    #     "k":3,
    #     "linear_digit":1,
    #     "node_features":256,
    #     "filters":4,
    #     "lr":0.01,
    #     "lags":256,
    #     "train_ratio":0.9
    # },
    # {
    #     "k":3,
    #     "linear_digit":1,
    #     "node_features":256,
    #     "filters":2,
    #     "lr":0.0025,
    #     "lags":256,
    #     "train_ratio":0.9
    # },
    # {
    #     "k":3,
    #     "linear_digit":1,
    #     "node_features":256,
    #     "filters":4,
    #     "lr":0.0025,
    #     "lags":256,
    #     "train_ratio":0.9
    # },
    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":256,
    #     "filters":2,
    #     "lr":0.01,
    #     "lags":256,
    #     "train_ratio":0.9
    # },
    {
        "k":2,
        "linear_digit":1,
        "node_features":256,
        "filters":4,
        "lr":0.0025,
        "lags":256,
        "train_ratio":0.9
    },
    {
        "k":2,
        "linear_digit":1,
        "node_features":256,
        "filters":4,
        "lr":0.005,
        "lags":256,
        "train_ratio":0.9
    },
    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":256,
    #     "filters":2,
    #     "lr":0.005,
    #     "lags":256,
    #     "train_ratio":0.9
    # },
    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":256,
    #     "filters":4,
    #     "lr":0.01,
    #     "lags":256,
    #     "train_ratio":0.9
    # },
]
print(tests)
print(len(tests))

for test in tests:
    print("====================================")
    print(test)
    dataset = loader.get_dataset(lags=test["lags"])
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=test["train_ratio"])
    try:
        startTime = time.time()
        #####script#####
        model = RecurrentGCN(node_features=test["node_features"], filters=test["filters"], k=test["k"], linear_digit=test["linear_digit"])
        optimizer = torch.optim.Adam(model.parameters(), lr=test["lr"])
        model.train()
        for epoch in tqdm(range(50)):
            for _time, snapshot in enumerate(train_dataset):
                y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
                cost = torch.mean((y_hat-snapshot.y)**2)
                cost.backward()
                optimizer.step()
                optimizer.zero_grad()
        model.eval()
        cost = 0
        for _time, snapshot in enumerate(test_dataset):
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            cost = cost + torch.mean((y_hat-snapshot.y)**2)
        cost = cost / (_time+1)
        cost = cost.item()
        # print("MSE: {:.4f}".format(cost))
        # print("MSE: {:.10f}".format(cost))
        print(cost)
        #####script#####
        executionTime = (time.time() - startTime)
        print('Execution time in seconds: ' + str(executionTime))
    except Exception as e:
        print("Error!\n", e)
