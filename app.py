# SET NUMBA_ENABLE_CUDASIM=1
from numba import cuda
print(cuda.gpus)
device_id = 0
cuda.select_device(device_id)

import torch
print(torch.cuda.is_available())

import tensorflow as tf
print(tf.test.gpu_device_name())

from torch_geometric_temporal.dataset import WikiMathsDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

loader = WikiMathsDatasetLoader()

dataset = loader.get_dataset(lags=14)

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.5)
print(train_dataset)
print(test_dataset)

import time
import torch
from tqdm import tqdm
import torch.nn.functional
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU

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
    {
        "k":2,
        "linear_digit":1,
        "node_features":14,
        "filters":32,
        "lr":0.01
    },
    # ====================================
    # {'k': 2, 'linear_digit': 1, 'node_features': 14, 'filters': 32, 'lr': 0.01}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [14:57<00:00, 17.95s/it]
    # MSE: 0.8014
    # MSE: 0.8013635278
    # 0.8013635277748108
    # Execution time in seconds: 912.6144635677338
    # ====================================

    {
        "k":3,
        "linear_digit":1,
        "node_features":14,
        "filters":32,
        "lr":0.01
    },
    # ====================================
    # {'k': 3, 'linear_digit': 1, 'node_features': 14, 'filters': 32, 'lr': 0.01}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [23:39<00:00, 28.39s/it]
    # MSE: 0.8164
    # MSE: 0.8163800836
    # 0.8163800835609436
    # Execution time in seconds: 1444.7062287330627


    {
        "k":4,
        "linear_digit":1,
        "node_features":14,
        "filters":32,
        "lr":0.01
    },
    # ====================================
    # {'k': 4, 'linear_digit': 1, 'node_features': 14, 'filters': 32, 'lr': 0.01}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [31:53<00:00, 38.27s/it]
    # MSE: 0.7932
    # MSE: 0.7932114601
    # 0.7932114601135254
    # Execution time in seconds: 1947.6272599697113
    # ====================================

    # {
    #     "k":5,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.01
    # },
    # {
    #     "k":10,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.01
    # },


    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":20,
    #     "filters":32,
    #     "lr":0.01
    # },
    # error
    # Traceback (most recent call last):
    #   File "C:\Users\MAX\pytorch_geometric_temporal\t.py", line 265, in <module>
    #     for _time, snapshot in enumerate(train_dataset):
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Users\MAX\pytorch_geometric_temporal\t.py", line 42, in forward
    #     def forward(self, x, edge_index, edge_weight):
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Users\MAX\pytorch_geometric_temporal\torch_geometric_temporal\nn\recurrent\gconv_gru.py", line 163, in forward
    #     Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
    #   File "C:\Users\MAX\pytorch_geometric_temporal\torch_geometric_temporal\nn\recurrent\gconv_gru.py", line 120, in _calculate_update_gate
    #     Z = self.conv_x_z(X, edge_index, edge_weight)
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Python310\lib\site-packages\torch_geometric\nn\conv\cheb_conv.py", line 145, in forward
    #     out = self.lins[0](Tx_0)
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Python310\lib\site-packages\torch_geometric\nn\dense\linear.py", line 118, in forward
    #     return F.linear(x, self.weight, self.bias)
    # RuntimeError: mat1 and mat2 shapes cannot be multiplied (1068x14 and 20x32)


    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":50,
    #     "filters":32,
    #     "lr":0.01
    # },
    # Traceback (most recent call last):
    #   File "C:\Users\MAX\pytorch_geometric_temporal\t.py", line 291, in <module>
    #     y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Users\MAX\pytorch_geometric_temporal\t.py", line 43, in forward
    #     h = self.recurrent(x, edge_index, edge_weight)
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Users\MAX\pytorch_geometric_temporal\torch_geometric_temporal\nn\recurrent\gconv_gru.py", line 163, in forward
    #     Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
    #   File "C:\Users\MAX\pytorch_geometric_temporal\torch_geometric_temporal\nn\recurrent\gconv_gru.py", line 120, in _calculate_update_gate
    #     Z = self.conv_x_z(X, edge_index, edge_weight)
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Python310\lib\site-packages\torch_geometric\nn\conv\cheb_conv.py", line 145, in forward
    #     out = self.lins[0](Tx_0)
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Python310\lib\site-packages\torch_geometric\nn\dense\linear.py", line 118, in forward
    #     return F.linear(x, self.weight, self.bias)
    # RuntimeError: mat1 and mat2 shapes cannot be multiplied (1068x14 and 50x32)

    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":50,
    #     "filters":64,
    #     "lr":0.01
    # },
    # Traceback (most recent call last):
    #   File "C:\Users\MAX\pytorch_geometric_temporal\t.py", line 314, in <module>
    #     y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Users\MAX\pytorch_geometric_temporal\t.py", line 43, in forward
    #     h = self.recurrent(x, edge_index, edge_weight)
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Users\MAX\pytorch_geometric_temporal\torch_geometric_temporal\nn\recurrent\gconv_gru.py", line 163, in forward
    #     Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
    #   File "C:\Users\MAX\pytorch_geometric_temporal\torch_geometric_temporal\nn\recurrent\gconv_gru.py", line 120, in _calculate_update_gate
    #     Z = self.conv_x_z(X, edge_index, edge_weight)
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Python310\lib\site-packages\torch_geometric\nn\conv\cheb_conv.py", line 145, in forward
    #     out = self.lins[0](Tx_0)
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Python310\lib\site-packages\torch_geometric\nn\dense\linear.py", line 118, in forward
    #     return F.linear(x, self.weight, self.bias)
    # RuntimeError: mat1 and mat2 shapes cannot be multiplied (1068x14 and 50x64)



    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":50,
    #     "filters":20,
    #     "lr":0.01
    # },
    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":50,
    #     "filters":100,
    #     "lr":0.01
    # },
    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":80,
    #     "filters":100,
    #     "lr":0.01
    # },
    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":80,
    #     "filters":40,
    #     "lr":0.01
    # },



    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.02
    # },
    # ====================================
    # {'k': 2, 'linear_digit': 1, 'node_features': 14, 'filters': 32, 'lr': 0.02}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [15:21<00:00, 18.43s/it]
    # MSE: 0.8355
    # MSE: 0.8355252743
    # 0.8355252742767334
    # Execution time in seconds: 936.6834886074066
    # ====================================



    # {
    #     "k":3,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.02
    # },
    # {'k': 3, 'linear_digit': 1, 'node_features': 14, 'filters': 32, 'lr': 0.02}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [30:15<00:00, 36.31s/it]
    # MSE: 0.8605
    # MSE: 0.8604558110
    # 0.8604558110237122
    # Execution time in seconds: 1839.261245727539


    # {
    #     "k":4,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.02
    # },
    # ====================================
    # {'k': 4, 'linear_digit': 1, 'node_features': 14, 'filters': 32, 'lr': 0.02}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [38:29<00:00, 46.20s/it]
    # MSE: 0.8616
    # MSE: 0.8616055846
    # 0.8616055846214294
    # Execution time in seconds: 2346.4781737327576
    # ====================================


    # {
    #     "k":5,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.02
    # },
    # ====================================
    # {'k': 5, 'linear_digit': 1, 'node_features': 14, 'filters': 32, 'lr': 0.02}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [41:57<00:00, 50.34s/it]
    # MSE: 0.8868
    # MSE: 0.8867608309
    # 0.8867608308792114
    # Execution time in seconds: 2559.8424396514893
    # ====================================



    # {
    #     "k":10,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.02
    # },
    # ====================================
    # {'k': 10, 'linear_digit': 1, 'node_features': 14, 'filters': 32, 'lr': 0.02}
    # 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [1:27:58<00:00, 105.56s/it]
    # MSE: 0.8465
    # MSE: 0.8464503288
    # 0.8464503288269043
    # Execution time in seconds: 5376.782430171967
    # ====================================

    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":20,
    #     "filters":32,
    #     "lr":0.02
    # },
    # Traceback (most recent call last):
    #   File "C:\Users\MAX\pytorch_geometric_temporal\t.py", line 339, in <module>
    #     "filters":20,
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Users\MAX\pytorch_geometric_temporal\t.py", line 43, in forward
    #     h = self.recurrent(x, edge_index, edge_weight)
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Users\MAX\pytorch_geometric_temporal\torch_geometric_temporal\nn\recurrent\gconv_gru.py", line 163, in forward
    #     Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
    #   File "C:\Users\MAX\pytorch_geometric_temporal\torch_geometric_temporal\nn\recurrent\gconv_gru.py", line 120, in _calculate_update_gate
    #     Z = self.conv_x_z(X, edge_index, edge_weight)
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Python310\lib\site-packages\torch_geometric\nn\conv\cheb_conv.py", line 145, in forward
    #     out = self.lins[0](Tx_0)
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Python310\lib\site-packages\torch_geometric\nn\dense\linear.py", line 118, in forward
    #     return F.linear(x, self.weight, self.bias)
    # RuntimeError: mat1 and mat2 shapes cannot be multiplied (1068x14 and 20x32)

    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":50,
    #     "filters":32,
    #     "lr":0.02
    # },
    # Traceback (most recent call last):
    #   File "C:\Users\MAX\pytorch_geometric_temporal\t.py", line 411, in <module>
    #     y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Users\MAX\pytorch_geometric_temporal\t.py", line 43, in forward
    #     h = self.recurrent(x, edge_index, edge_weight)
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Users\MAX\pytorch_geometric_temporal\torch_geometric_temporal\nn\recurrent\gconv_gru.py", line 163, in forward
    #     Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
    #   File "C:\Users\MAX\pytorch_geometric_temporal\torch_geometric_temporal\nn\recurrent\gconv_gru.py", line 120, in _calculate_update_gate
    #     Z = self.conv_x_z(X, edge_index, edge_weight)
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Python310\lib\site-packages\torch_geometric\nn\conv\cheb_conv.py", line 145, in forward
    #     out = self.lins[0](Tx_0)
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Python310\lib\site-packages\torch_geometric\nn\dense\linear.py", line 118, in forward
    #     return F.linear(x, self.weight, self.bias)
    # RuntimeError: mat1 and mat2 shapes cannot be multiplied (1068x14 and 50x32)

    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":50,
    #     "filters":64,
    #     "lr":0.02
    # },
    # Traceback (most recent call last):
    #   File "C:\Users\MAX\pytorch_geometric_temporal\t.py", line 434, in <module>
    #     y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Users\MAX\pytorch_geometric_temporal\t.py", line 43, in forward
    #     h = self.recurrent(x, edge_index, edge_weight)
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Users\MAX\pytorch_geometric_temporal\torch_geometric_temporal\nn\recurrent\gconv_gru.py", line 163, in forward
    #     Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
    #   File "C:\Users\MAX\pytorch_geometric_temporal\torch_geometric_temporal\nn\recurrent\gconv_gru.py", line 120, in _calculate_update_gate
    #     Z = self.conv_x_z(X, edge_index, edge_weight)
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Python310\lib\site-packages\torch_geometric\nn\conv\cheb_conv.py", line 145, in forward
    #     out = self.lins[0](Tx_0)
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Python310\lib\site-packages\torch_geometric\nn\dense\linear.py", line 118, in forward
    #     return F.linear(x, self.weight, self.bias)
    # RuntimeError: mat1 and mat2 shapes cannot be multiplied (1068x14 and 50x64)



    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":50,
    #     "filters":20,
    #     "lr":0.02
    # },
    # Traceback (most recent call last):
    #   File "C:\Users\MAX\pytorch_geometric_temporal\t.py", line 459, in <module>
    #     y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Users\MAX\pytorch_geometric_temporal\t.py", line 43, in forward
    #     h = self.recurrent(x, edge_index, edge_weight)
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Users\MAX\pytorch_geometric_temporal\torch_geometric_temporal\nn\recurrent\gconv_gru.py", line 163, in forward
    #     Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
    #   File "C:\Users\MAX\pytorch_geometric_temporal\torch_geometric_temporal\nn\recurrent\gconv_gru.py", line 120, in _calculate_update_gate
    #     Z = self.conv_x_z(X, edge_index, edge_weight)
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Python310\lib\site-packages\torch_geometric\nn\conv\cheb_conv.py", line 145, in forward
    #     out = self.lins[0](Tx_0)
    #   File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    #     return forward_call(*input, **kwargs)
    #   File "C:\Python310\lib\site-packages\torch_geometric\nn\dense\linear.py", line 118, in forward
    #     return F.linear(x, self.weight, self.bias)
    # RuntimeError: mat1 and mat2 shapes cannot be multiplied (1068x14 and 50x20)


    {
        "k":2,
        "linear_digit":1,
        "node_features":50,
        "filters":100,
        "lr":0.02
    },
    # raceback (most recent call last):
    # File "C:\Users\MAX\pytorch_geometric_temporal\t.py", line 483, in <module>
    # y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    # File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    # return forward_call(*input, **kwargs)
    # File "C:\Users\MAX\pytorch_geometric_temporal\t.py", line 43, in forward
    # h = self.recurrent(x, edge_index, edge_weight)
    # File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    # return forward_call(*input, **kwargs)
    # File "C:\Users\MAX\pytorch_geometric_temporal\torch_geometric_temporal\nn\recurrent\gconv_gru.py", line 163, in forward
    # Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
    # File "C:\Users\MAX\pytorch_geometric_temporal\torch_geometric_temporal\nn\recurrent\gconv_gru.py", line 120, in _calculate_update_gate
    # Z = self.conv_x_z(X, edge_index, edge_weight)
    # File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    # return forward_call(*input, **kwargs)
    # File "C:\Python310\lib\site-packages\torch_geometric\nn\conv\cheb_conv.py", line 145, in forward
    # out = self.lins[0](Tx_0)
    # File "C:\Python310\lib\site-packages\torch\nn\modules\module.py", line 1110, in _call_impl
    # return forward_call(*input, **kwargs)
    # File "C:\Python310\lib\site-packages\torch_geometric\nn\dense\linear.py", line 118, in forward
    # return F.linear(x, self.weight, self.bias)
    # RuntimeError: mat1 and mat2 shapes cannot be multiplied (1068x14 and 50x100)


    {
        "k":2,
        "linear_digit":1,
        "node_features":80,
        "filters":100,
        "lr":0.02
    },
    {
        "k":2,
        "linear_digit":1,
        "node_features":80,
        "filters":40,
        "lr":0.02
    },


]
print(tests)
print(len(tests))

for test in tests:
    print("====================================")
    print(test)
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
    print("MSE: {:.4f}".format(cost))
    print("MSE: {:.10f}".format(cost))
    print(cost)
    #####script#####
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))
