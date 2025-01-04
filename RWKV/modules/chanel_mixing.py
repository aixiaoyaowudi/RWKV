import torch
import torch.nn as nn

class RWKV_Channel_Mixing_v4(nn.Module):

    def __init__(self, init_weights : bool, layer_index : int, total_layers : int, emb_size : int, channel_mixing_weight_std : float = 2, hidden_size : int = None):

        super(RWKV_Channel_Mixing_v4, self).__init__()

        if hidden_size == None:
            hidden_size = 4 * emb_size # 4 * emb_size is used in official training

        self.receptance_trans = nn.Linear(emb_size, emb_size, bias = False)
        self.key_trans = nn.Linear(emb_size, hidden_size, bias = False)

        self.value_trans = nn.Linear(hidden_size, emb_size, bias = False)
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        if init_weights:
            
            with torch.no_grad():
                self.mu_receptance = nn.Parameter(torch.pow(torch.arange(0, emb_size) / emb_size, 1 - layer_index / total_layers))
                self.mu_key = nn.Parameter(self.mu_receptance.data.clone().detach())
                self.receptance_trans.weight.zero_()
                self.key_trans.weight.zero_()
                nn.init.normal_(self.value_trans.weight, std = channel_mixing_weight_std)
        
        else:
            self.mu_receptance = nn.Parameter(torch.empty((emb_size, )))
            self.mu_key = nn.Parameter(torch.empty((emb_size, )))
    
    def forward(self, x):
        shifted_x = self.time_shift(x)
        receptance = self.receptance_trans(self.mu_receptance * x + (1 - self.mu_receptance) * shifted_x)
        key = self.key_trans(self.mu_key * x + (1 - self.mu_key) * shifted_x)
        value = self.value_trans(key)

        return torch.sigmoid(receptance) * value