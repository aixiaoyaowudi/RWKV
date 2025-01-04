import torch
import torch.nn as nn

from .time_mixing import RWKV_Time_Mixing_v4
from .chanel_mixing import RWKV_Channel_Mixing_v4
import lightning as L
import torchmetrics

class auxiliary_loss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, output):
        ctx.save_for_backward(output)
        return loss
    @staticmethod
    def backward(ctx, grad_loss):
        output = ctx.saved_tensors[0]
        factor = 1e-4 / (output.shape[0] * output.shape[1])
        output_max, indices = torch.max(output, -1, keepdim=True)
        grad_output = torch.zeros_like(output)
        grad_output.scatter_(-1, indices, output_max * factor)
        return (grad_loss, grad_output)

class RWKV_block_v4(nn.Module):

    def __init__(self, init_weights : bool, layer_index : int, total_layers : int, emb_size : int,
                 time_mixing_weight_std : float = 2, time_mixing_hidden_size : int = None,
                 channel_mixing_weight_std : float = 2, channel_mixing_hidden_size : int = None):
        super(RWKV_block_v4, self).__init__()
        self.norm_time_mixing = nn.LayerNorm(emb_size)
        self.norm_channel_mixing = nn.LayerNorm(emb_size)
        self.time_mixing = RWKV_Time_Mixing_v4(init_weights, layer_index, total_layers, emb_size, time_mixing_weight_std, time_mixing_hidden_size)
        self.channel_mixing = RWKV_Channel_Mixing_v4(init_weights, layer_index, total_layers, emb_size, channel_mixing_weight_std, channel_mixing_hidden_size)
    def forward(self, x):
        x = x + self.time_mixing(self.norm_time_mixing(x))
        x = x + self.channel_mixing(self.norm_channel_mixing(x))
        return x

class RWKV_v4(L.LightningModule):

    def __init__(self, init_weights : bool, vocab_size : int, total_layers : int, emb_size : int,
                 time_mixing_weight_std : float = 2, time_mixing_hidden_size : int = None,
                 channel_mixing_weight_std : float = 2, channel_mixing_hidden_size : int = None, embedding_init_value : float = 1e-4,
                 adam_betas = (0.9, 0.99), learning_rate = 6e-4):

        # super(RWKV_v4, self).__init__()
        
        super().__init__()

        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Embedding(vocab_size, emb_size),
            nn.LayerNorm(emb_size),
            *[RWKV_block_v4(init_weights, layer_index, total_layers, emb_size,
                            time_mixing_weight_std, time_mixing_hidden_size,
                            channel_mixing_weight_std, channel_mixing_hidden_size) for layer_index in range(total_layers)],
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, vocab_size, bias = False),
            nn.Softmax(dim = 1)
        )
        self.accuracy_func = torchmetrics.Accuracy(task = 'multiclass', num_classes = vocab_size)

        if init_weights:
            with torch.no_grad():
                nn.init.uniform_(self.model[0].weight, - embedding_init_value, embedding_init_value)
        
        self.adam_betas = adam_betas
        self.learning_rate = learning_rate
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        print(y_hat.shape, y.shape)
        loss = auxiliary_loss.apply(torch.nn.functional.cross_entropy(y_hat.view(-1, y_hat.shape[-1]), y.flatten()), y_hat)
        acc = self.accuracy_func(y, torch.max(y_hat.detach(), -1)[1])
        self.log('training_loss', loss, on_step = True, on_epoch = True, prog_bar = True, logger=True)
        self.log('training_acc', acc, on_step  = True, on_epoch = True, prog_bar = True, logger = True)
        return {'loss': loss, 'acc': acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), betas = self.adam_betas, lr = self.learning_rate)