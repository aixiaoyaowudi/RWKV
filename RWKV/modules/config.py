# import this file before importing any other files

from math import sqrt

context_len = 1024

vocab_size = 50277

emb_size = 768

total_layers = 12

time_mixing_hidden_size = emb_size

channel_mixing_hidden_size = 4 * emb_size

time_mixing_weight_std = sqrt(channel_mixing_hidden_size / time_mixing_hidden_size)

channel_mixing_weight_std = time_mixing_weight_std

adam_betas = (0.9, 0.99)

miniepoch_size = 768

epoch_count = 8192

learning_rate = 6e-4

batch_size = 8

gpu_count = 2