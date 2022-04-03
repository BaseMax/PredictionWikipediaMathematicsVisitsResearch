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































    # ENDDDDDDDDDDDDDDDDDDDDDDDD


    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":14*4,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14*4,
    #     "train_ratio":0.5
    # },
    # {'k': 2, 'linear_digit': 1, 'node_features': 56, 'filters': 32, 'lr': 0.01, 'lags': 56, 'train_ratio': 0.5}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [21:13<00:00, 25.46s/it]
    # MSE: 0.8365
    # MSE: 0.8364545107
    # 0.8364545106887817
    # Execution time in seconds: 1296.4120910167694

    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":14*5,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14*5,
    #     "train_ratio":0.5
    # },
    # {'k': 2, 'linear_digit': 1, 'node_features': 70, 'filters': 32, 'lr': 0.01, 'lags': 70, 'train_ratio': 0.5}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [22:14<00:00, 26.70s/it]
    # MSE: 0.8788
    # MSE: 0.8788001537
    # 0.8788001537322998
    # Execution time in seconds: 1358.8758063316345
    
    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":14*6,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14*6,
    #     "train_ratio":0.5
    # },
    # {'k': 2, 'linear_digit': 1, 'node_features': 84, 'filters': 32, 'lr': 0.01, 'lags': 84, 'train_ratio': 0.5}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [19:25<00:00, 23.30s/it]
    # MSE: 0.9006
    # MSE: 0.9005643129
    # 0.9005643129348755
    # Execution time in seconds: 1185.7476696968079

    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":14*7,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14*7,
    #     "train_ratio":0.5
    # },
    # {'k': 2, 'linear_digit': 1, 'node_features': 98, 'filters': 32, 'lr': 0.01, 'lags': 98, 'train_ratio': 0.5}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [19:56<00:00, 23.92s/it]
    # MSE: 0.8544
    # MSE: 0.8543722630
    # 0.8543722629547119
    # Execution time in seconds: 1216.9463980197906

    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":14*3,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14*3,
    #     "train_ratio":0.5
    # },
    # {'k': 2, 'linear_digit': 1, 'node_features': 42, 'filters': 32, 'lr': 0.01, 'lags': 42, 'train_ratio': 0.5}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [18:16<00:00, 21.93s/it]
    # MSE: 0.8399
    # MSE: 0.8399303555
    # 0.8399303555488586
    # Execution time in seconds: 1114.2216057777405
    # ====================================
    
    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":14*2,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14*2,
    #     "train_ratio":0.5
    # },
    # ====================================
    # {'k': 2, 'linear_digit': 1, 'node_features': 28, 'filters': 32, 'lr': 0.01, 'lags': 28, 'train_ratio': 0.5}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [17:11<00:00, 20.63s/it]
    # MSE: 0.8465
    # MSE: 0.8465337753
    # 0.8465337753295898
    # Execution time in seconds: 1050.6477363109589
    # ====================================

    # ENDDDDDDDDDDDDDDDDDDDD






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
    # {'k': 1, 'linear_digit': 1, 'node_features': 56, 'filters': 32, 'lr': 0.01, 'lags': 56, 'train_ratio': 0.5}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [07:33<00:00,  9.06s/it]
    # MSE: 0.8561
    # MSE: 0.8561053872
    # 0.856105387210846
    # Execution time in seconds: 461.94866609573364

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":14*5,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14*5,
    #     "train_ratio":0.5
    # },
    # {'k': 1, 'linear_digit': 1, 'node_features': 70, 'filters': 32, 'lr': 0.01, 'lags': 70, 'train_ratio': 0.5}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [08:16<00:00,  9.93s/it]
    # MSE: 0.8763
    # MSE: 0.8762531281
    # 0.8762531280517578
    # Execution time in seconds: 505.11158537864685

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":14*6,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14*6,
    #     "train_ratio":0.5
    # },
    # {'k': 1, 'linear_digit': 1, 'node_features': 84, 'filters': 32, 'lr': 0.01, 'lags': 84, 'train_ratio': 0.5}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [08:40<00:00, 10.42s/it]
    # MSE: 0.9410
    # MSE: 0.9409999847
    # 0.9409999847412109
    # Execution time in seconds: 529.0221219062805

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":14*7,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14*7,
    #     "train_ratio":0.5
    # },
    # {'k': 1, 'linear_digit': 1, 'node_features': 98, 'filters': 32, 'lr': 0.01, 'lags': 98, 'train_ratio': 0.5}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [08:59<00:00, 10.79s/it]
    # MSE: 0.9204
    # MSE: 0.9203919768
    # 0.9203919768333435
    # Execution time in seconds: 547.9380512237549

    
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":14*3,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.5
    # },
    # {'k': 1, 'linear_digit': 1, 'node_features': 42, 'filters': 32, 'lr': 0.01}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [07:16<00:00,  8.72s/it]
    # MSE: 0.8508
    # MSE: 0.8508368134
    # 0.8508368134498596
    # Execution time in seconds: 443.01936984062195

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":14*2,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.5
    # },
    # {'k': 1, 'linear_digit': 1, 'node_features': 28, 'filters': 32, 'lr': 0.01}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [07:14<00:00,  8.68s/it]
    # MSE: 0.8761
    # MSE: 0.8761430383
    # 0.8761430382728577
    # Execution time in seconds: 441.0587899684906

    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.5
    # },
    # ====================================
    # {'k': 2, 'linear_digit': 1, 'node_features': 14, 'filters': 32, 'lr': 0.01}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [14:56<00:00, 17.93s/it]
    # MSE: 0.8143
    # MSE: 0.8143236637
    # 0.8143236637115479
    # Execution time in seconds: 911.0427641868591
    # ====================================

    # {
    #     "k":3,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.5
    # },
    # ====================================
    # {'k': 3, 'linear_digit': 1, 'node_features': 14, 'filters': 32, 'lr': 0.01}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [23:39<00:00, 28.39s/it]
    # MSE: 0.8164
    # MSE: 0.8163800836
    # 0.8163800835609436
    # Execution time in seconds: 1444.7062287330627

    # {
    #     "k":4,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.5
    # },
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
    #     "node_features":20,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.5
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
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.5
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
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.5
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
    #     "lr":0.02,
    #     "lags":14,
    #     "train_ratio":0.5
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
    #     "lr":0.02,
    #     "lags":14,
    #     "train_ratio":0.5
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
    #     "lr":0.02,
    #     "lags":14,
    #     "train_ratio":0.5
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
    #     "lr":0.02,
    #     "lags":14,
    #     "train_ratio":0.5
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
    #     "lr":0.02,
    #     "lags":14,
    #     "train_ratio":0.5
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
    #     "lr":0.02,
    #     "lags":14,
    #     "train_ratio":0.5
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
    #     "lr":0.02,
    #     "lags":14,
    #     "train_ratio":0.5
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
    #     "lr":0.02,
    #     "lags":14,
    #     "train_ratio":0.5
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


    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":50,
    #     "filters":100,
    #     "lr":0.02,
    #     "lags":14,
    #     "train_ratio":0.5
    # },
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
]

tests = [
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":50,
    #     "lr":0.02,
    #     "lags":14,
    #     "train_ratio":0.5
    # },
    # {'k': 1, 'linear_digit': 1, 'node_features': 14, 'filters': 50, 'lr': 0.02, 'lags': 14, 'train_ratio': 0.5}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [07:37<00:00,  9.16s/it]
    # MSE: 0.8964
    # MSE: 0.8963724375
    # 0.8963724374771118
    # Execution time in seconds: 464.7623805999756
    # ====================================

    # dr behzad asked me....
    # lags = 16,32,64,128,256, k=1 , lin_dig = 1, node_feature = 16, filter = 16 , lr = 0.01
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":16,
    #     "train_ratio":0.7
    # },
    # {'k': 1, 'linear_digit': 1, 'node_features': 16, 'filters': 16, 'lr': 0.01, 'lags': 16, 'train_ratio': 0.7}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [10:04<00:00, 12.09s/it]
    # MSE: 1.4011
    # MSE: 1.4011325836
    # 1.401132583618164
    # Execution time in seconds: 608.520295381546
    # ====================================

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":32,
    #     "train_ratio":0.7
    # },
    # RuntimeError: mat1 and mat2 shapes cannot be multiplied (1068x32 and 16x16)

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":64,
    #     "train_ratio":0.7
    # },
    # RuntimeError: mat1 and mat2 shapes cannot be multiplied (1068x64 and 16x16)

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":128,
    #     "train_ratio":0.7
    # },
    # RuntimeError: mat1 and mat2 shapes cannot be multiplied (1068x128 and 16x16)

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":256,
    #     "train_ratio":0.7
    # },
    # RuntimeError: mat1 and mat2 shapes cannot be multiplied (1068x256 and 16x16)

    # lags = 32, k=1,2,3 , lin_dig = 1, node_feature = 16, filter = 16 , lr = 0.01
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":32,
    #     "train_ratio":0.7
    # },
    # Error

    # {
    #     "k":2,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":32,
    #     "train_ratio":0.7
    # },
    # Error

    # {
    #     "k":3,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":32,
    #     "train_ratio":0.7
    # },
    # Error

    # lags = 32, k=1 , lin_dig = 1, node_feature = 2,4,8,16,32, filter = 16 , lr = 0.01
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":2,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":32,
    #     "train_ratio":0.7
    # },
    # Error

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":4,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":32,
    #     "train_ratio":0.7
    # },
    # Error

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":8,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":32,
    #     "train_ratio":0.7
    # },
    # Error

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":32,
    #     "train_ratio":0.7
    # },
    # Error

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":32,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":32,
    #     "train_ratio":0.7
    # },
    # {'k': 1, 'linear_digit': 1, 'node_features': 32, 'filters': 16, 'lr': 0.01, 'lags': 32, 'train_ratio': 0.7}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [10:03<00:00, 12.08s/it]
    # MSE: 1.6347
    # MSE: 1.6346751451
    # 1.634675145149231
    # Execution time in seconds: 607.8879504203796

    # lags = 32, k=1 , lin_dig = 1, node_feature = 16, filter = 2,4,8,16,32 , lr = 0.01
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":2,
    #     "lr":0.01,
    #     "lags":32,
    #     "train_ratio":0.7
    # },
    # Error

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":4,
    #     "lr":0.01,
    #     "lags":32,
    #     "train_ratio":0.7
    # },
    # Error

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":8,
    #     "lr":0.01,
    #     "lags":32,
    #     "train_ratio":0.7
    # },
    # Error

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":32,
    #     "train_ratio":0.7
    # },
    # Error

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":32,
    #     "train_ratio":0.7
    # },
    # Error

    # lags = 32, k=1 , lin_dig = 1, node_feature = 16, filter = 16, lr = 0.01, 0.02, 0.03
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":32,
    #     "train_ratio":0.7
    # },
    # Error

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":32,
    #     "lr":0.02,
    #     "lags":32,
    #     "train_ratio":0.7
    # },
    # Error

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":32,
    #     "lr":0.03,
    #     "lags":32,
    #     "train_ratio":0.7
    # },
    # Error

    # lags = 32, k=1 , lin_dig = 1,2,3, node_feature = 16, filter = 16 , lr = 0.01
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":32,
    #     "train_ratio":0.7
    # },
    # Error

    # {
    #     "k":1,
    #     "linear_digit":2,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":32,
    #     "train_ratio":0.7
    # },
    # Error

    # {
    #     "k":1,
    #     "linear_digit":3,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":32,
    #     "train_ratio":0.7
    # },
    # Error




    # My change refer to dr. behzad tests
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

    # lags = 32, k=1,2,3 , lin_dig = 1, node_feature = 16, filter = 16 , lr = 0.01
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

    # lags = 32, k=1 , lin_dig = 1, node_feature = 2,4,8,16,32, filter = 16 , lr = 0.01
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

    # lags = 32, k=1 , lin_dig = 1, node_feature = 16, filter = 2,4,8,16,32 , lr = 0.01
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

    # lags = 32, k=1 , lin_dig = 1, node_feature = 16, filter = 16, lr = 0.01, 0.02, 0.03
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

    # lags = 32, k=1 , lin_dig = 1,2,3, node_feature = 16, filter = 16 , lr = 0.01
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


]

tests = [
    # e = 7, lags = 32, k=1 , lin_dig = 1, node_feature = 16, filter = 16 , lr = 0.01, train_test = 0.5,0.7,0.9
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.3
    # },
    # {'k': 1, 'linear_digit': 1, 'node_features': 14, 'filters': 32, 'lr': 0.01, 'lags': 14, 'train_ratio': 0.3}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [04:19<00:00,  5.19s/it]
    # MSE: 1.0906
    # MSE: 1.0906275511
    # 1.0906275510787964
    # Execution time in seconds: 268.4892053604126
    # ====================================

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.4
    # },
    # {'k': 1, 'linear_digit': 1, 'node_features': 14, 'filters': 32, 'lr': 0.01, 'lags': 14, 'train_ratio': 0.4}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [05:54<00:00,  7.09s/it]
    # MSE: 0.8775
    # MSE: 0.8774722815
    # 0.8774722814559937
    # Execution time in seconds: 362.50277638435364
    # ====================================

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.6
    # },
    # {'k': 1, 'linear_digit': 1, 'node_features': 14, 'filters': 32, 'lr': 0.01, 'lags': 14, 'train_ratio': 0.6}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [08:47<00:00, 10.55s/it]
    # MSE: 0.8744
    # MSE: 0.8744056225
    # 0.8744056224822998
    # Execution time in seconds: 532.7825939655304
    # ====================================

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.7
    # },
    # {'k': 1, 'linear_digit': 1, 'node_features': 14, 'filters': 32, 'lr': 0.01, 'lags': 14, 'train_ratio': 0.7}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [10:28<00:00, 12.56s/it]
    # MSE: 1.3145
    # MSE: 1.3144520521
    # 1.314452052116394
    # Execution time in seconds: 632.4220359325409

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":14,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":14,
    #     "train_ratio":0.9
    # },
    # {'k': 1, 'linear_digit': 1, 'node_features': 14, 'filters': 32, 'lr': 0.01, 'lags': 14, 'train_ratio': 0.9}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [13:02<00:00, 15.65s/it]
    # MSE: 0.6677
    # MSE: 0.6676675677
    # 0.66766756772995
    # Execution time in seconds: 783.8723146915436

    # e = 7, lags = 32, k=1 , lin_dig = 1, node_feature = 16, filter = 16 , lr = 0.01, train_test = 0.5,0.7,0.9
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":32,
    #     "train_ratio":0.5
    # },
    # Error
    # mat1 and mat2 shapes cannot be multiplied (1068x32 and 16x16)

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":32,
    #     "train_ratio":0.7
    # },
    # Error
    # mat1 and mat2 shapes cannot be multiplied (1068x32 and 16x16)

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":32,
    #     "train_ratio":0.9
    # },
    # Error
    # mat1 and mat2 shapes cannot be multiplied (1068x32 and 16x16)
]

tests = [
    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":16,
    #     "train_ratio":0.3
    # },
    # {'k': 1, 'linear_digit': 1, 'node_features': 16, 'filters': 16, 'lr': 0.01, 'lags': 16, 'train_ratio': 0.3}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [04:22<00:00,  5.24s/it]
    # MSE: 1.0896
    # MSE: 1.0896383524
    # 1.089638352394104
    # Execution time in seconds: 271.48100447654724

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":16,
    #     "train_ratio":0.4
    # },
    # {'k': 1, 'linear_digit': 1, 'node_features': 16, 'filters': 16, 'lr': 0.01, 'lags': 16, 'train_ratio': 0.4}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [05:37<00:00,  6.75s/it]
    # MSE: 0.8601
    # MSE: 0.8601189256
    # 0.8601189255714417
    # Execution time in seconds: 345.3885200023651

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":16,
    #     "train_ratio":0.5
    # },
    # {'k': 1, 'linear_digit': 1, 'node_features': 16, 'filters': 16, 'lr': 0.01, 'lags': 16, 'train_ratio': 0.5}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [06:52<00:00,  8.26s/it]
    # MSE: 0.8373
    # MSE: 0.8372963071
    # 0.8372963070869446
    # Execution time in seconds: 419.25942969322205

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":16,
    #     "train_ratio":0.6
    # },
    # {'k': 1, 'linear_digit': 1, 'node_features': 16, 'filters': 16, 'lr': 0.01, 'lags': 16, 'train_ratio': 0.6}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [08:32<00:00, 10.25s/it]
    # MSE: 0.8801
    # MSE: 0.8800567985
    # 0.8800567984580994
    # Execution time in seconds: 517.7016937732697

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":16,
    #     "train_ratio":0.7
    # },
    # {'k': 1, 'linear_digit': 1, 'node_features': 16, 'filters': 16, 'lr': 0.01, 'lags': 16, 'train_ratio': 0.7}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [09:56<00:00, 11.92s/it]
    # MSE: 1.3648
    # MSE: 1.3647654057
    # 1.3647654056549072
    # Execution time in seconds: 600.0506024360657

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":16,
    #     "train_ratio":0.8
    # },
    # 677s
    # 0.8518020510673523

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":16,
    #     "train_ratio":0.9
    # },
    # {'k': 1, 'linear_digit': 1, 'node_features': 16, 'filters': 16, 'lr': 0.01, 'lags': 16, 'train_ratio': 0.9}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [12:59<00:00, 15.60s/it]
    # MSE: 0.6800
    # MSE: 0.6800107956
    # 0.6800107955932617
    # Execution time in seconds: 781.1437165737152

]
tests = [
    # new requests, time is 5:31am

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":16,
    #     "filters":64,
    #     "lr":0.01,
    #     "lags":16,
    #     "train_ratio":0.7
    # },
    # {'k': 1, 'linear_digit': 1, 'node_features': 16, 'filters': 64, 'lr': 0.01, 'lags': 16, 'train_ratio': 0.7}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [11:05<00:00, 13.31s/it]
    # MSE: 1.4078
    # MSE: 1.4077645540
    # 1.4077645540237427
    # Execution time in seconds: 671.2274804115295

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":64,
    #     "filters":2,
    #     "lr":0.01,
    #     "lags":64,
    #     "train_ratio":0.7
    # },
    # {'k': 1, 'linear_digit': 1, 'node_features': 64, 'filters': 2, 'lr': 0.01, 'lags': 64, 'train_ratio': 0.7}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [10:39<00:00, 12.78s/it]
    # MSE: 1.4670
    # MSE: 1.4670355320
    # 1.4670355319976807
    # Execution time in seconds: 643.2240812778473

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":64,
    #     "filters":4,
    #     "lr":0.01,
    #     "lags":64,
    #     "train_ratio":0.7
    # },
    # {'k': 1, 'linear_digit': 1, 'node_features': 64, 'filters': 4, 'lr': 0.01, 'lags': 64, 'train_ratio': 0.7}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [10:16<00:00, 12.32s/it]
    # MSE: 1.1776
    # MSE: 1.1776090860
    # 1.1776090860366821
    # Execution time in seconds: 620.5585584640503

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":64,
    #     "filters":8,
    #     "lr":0.01,
    #     "lags":64,
    #     "train_ratio":0.7
    # },
    # {'k': 1, 'linear_digit': 1, 'node_features': 64, 'filters': 8, 'lr': 0.01, 'lags': 64, 'train_ratio': 0.7}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [09:57<00:00, 11.95s/it]
    # MSE: 1.4935
    # MSE: 1.4935400486
    # 1.4935400485992432
    # Execution time in seconds: 601.4788415431976

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":64,
    #     "filters":16,
    #     "lr":0.01,
    #     "lags":64,
    #     "train_ratio":0.7
    # },
    # {'k': 1, 'linear_digit': 1, 'node_features': 64, 'filters': 16, 'lr': 0.01, 'lags': 64, 'train_ratio': 0.7}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [09:53<00:00, 11.87s/it]
    # MSE: 1.6202
    # MSE: 1.6202009916
    # 1.6202009916305542
    # Execution time in seconds: 597.5880570411682

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":64,
    #     "filters":32,
    #     "lr":0.01,
    #     "lags":64,
    #     "train_ratio":0.7
    # },
    # {'k': 1, 'linear_digit': 1, 'node_features': 64, 'filters': 32, 'lr': 0.01, 'lags': 64, 'train_ratio': 0.7}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [10:15<00:00, 12.31s/it]
    # MSE: 1.5204
    # MSE: 1.5203751326
    # 1.52037513256073
    # Execution time in seconds: 619.6944081783295

    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":64,
    #     "filters":64,
    #     "lr":0.01,
    #     "lags":64,
    #     "train_ratio":0.7
    # },    
    # {'k': 1, 'linear_digit': 1, 'node_features': 64, 'filters': 64, 'lr': 0.01, 'lags': 64, 'train_ratio': 0.7}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [10:48<00:00, 12.97s/it]
    # MSE: 1.5570
    # MSE: 1.5570083857
    # 1.5570083856582642
    # Execution time in seconds: 653.0965754985809


    # {
    #     "k":1,
    #     "linear_digit":1,
    #     "node_features":64,
    #     "filters":128,
    #     "lr":0.01,
    #     "lags":64,
    #     "train_ratio":0.7
    # },
    # {'k': 1, 'linear_digit': 1, 'node_features': 64, 'filters': 128, 'lr': 0.01, 'lags': 64, 'train_ratio': 0.7}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [11:33<00:00, 13.86s/it]
    # MSE: 1.5269
    # MSE: 1.5268894434
    # 1.526889443397522
    # Execution time in seconds: 697.7442462444305
    # ====================================






    # {
    #     "k":3,
    #     "linear_digit":1,
    #     "node_features":256,
    #     "filters":2,
    #     "lr":0.01,
    #     "lags":256,
    #     "train_ratio":0.9
    # },
    # {'k': 3, 'linear_digit': 1, 'node_features': 256, 'filters': 2, 'lr': 0.01, 'lags': 256, 'train_ratio': 0.9}
    # 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [1:20:57<00:00, 97.15s/it]
    # MSE: 0.6845
    # MSE: 0.6845269799
    # 0.6845269799232483
    # Execution time in seconds: 4868.179775953293

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
    # {'k': 2, 'linear_digit': 1, 'node_features': 256, 'filters': 2, 'lr': 0.01, 'lags': 256, 'train_ratio': 0.9}
    # 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [47:02<00:00, 56.45s/it]
    # MSE: 0.6924
    # MSE: 0.6923630834
    # 0.6923630833625793
    # Execution time in seconds: 2828.852452993393

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
    # dataset = loader.get_dataset(lags=14)
    dataset = loader.get_dataset(lags=test["lags"])
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=test["train_ratio"])
    # print(train_dataset)
    # print(test_dataset)

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
        print("MSE: {:.4f}".format(cost))
        print("MSE: {:.10f}".format(cost))
        print(cost)
        #####script#####
        executionTime = (time.time() - startTime)
        print('Execution time in seconds: ' + str(executionTime))
    except Exception as e:
        print("Error!\n", e)
