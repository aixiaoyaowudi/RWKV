import torch
import torch.nn as nn
from .wkv import WKV_v4
from math import log

class RWKV_Time_Mixing_v4(nn.Module):

    def __init__(self, init_weights : bool, layer_index : int, total_layers : int, emb_size : int, time_mixing_weight_std : float = 2, hidden_size : int = None):

        super(RWKV_Time_Mixing_v4, self).__init__()
        
        if hidden_size == None:
            hidden_size = emb_size

        self.receptance_trans = nn.Linear(emb_size, hidden_size, bias = False)
        self.key_trans = nn.Linear(emb_size, hidden_size, bias = False)
        self.value_trans = nn.Linear(emb_size, hidden_size, bias = False)

        self.out_trans = nn.Linear(hidden_size, emb_size, bias = False)
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        if init_weights:
            
            with torch.no_grad():
                
                self.mu_key = nn.Parameter(torch.pow(torch.arange(0, emb_size) / emb_size, 1 - layer_index / total_layers))
                self.mu_receptance = nn.Parameter(self.mu_key.data / 2)
                self.mu_value = nn.Parameter(self.mu_key.data + 0.3 * layer_index / (total_layers - 1))
                self.wkv_w = nn.Parameter(-5 + 8 * torch.pow(torch.arange(0, hidden_size) / (hidden_size - 1), 0.7 + 1.3 * layer_index / (total_layers - 1)))
                self.wkv_u = nn.Parameter(0.5 * (torch.arange(1, hidden_size + 1) % 3 - 1) + log(0.3))
                self.receptance_trans.weight.data.zero_()
                self.key_trans.weight.data.zero_()
                self.value_trans.weight.data.zero_()
                nn.init.normal_(self.out_trans.weight, std = time_mixing_weight_std)
        
        else:
            self.mu_receptance = nn.Parameter(torch.empty((emb_size, )))
            self.mu_key = nn.Parameter(torch.empty((emb_size, )))
            self.mu_value = nn.Parameter(torch.empty((emb_size, )))

            self.wkv_w = nn.Parameter(torch.empty((hidden_size, )))
            self.wkv_u = nn.Parameter(torch.empty((hidden_size, )))

    
    def forward(self, x):
        shifted_x = self.time_shift(x)
        receptance = self.receptance_trans(self.mu_receptance * x + (1 - self.mu_receptance) * shifted_x)
        key = self.key_trans(self.mu_key * x + (1 - self.mu_key) * shifted_x)
        value = self.value_trans(self.mu_value * x + (1 - self.mu_value) * shifted_x)

        wkv = WKV_v4.apply(key, value, self.wkv_w, self.wkv_u)
        # wkv = key
        return self.out_trans(torch.sigmoid(receptance) * wkv)


        
