from modules.config import *

from modules.model import RWKV_v4
import argparse
import os
import sys
import torch
from utils import BinIdxDataset
import random
import numpy as np
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import PreTrainedTokenizerFast

best_checkpoint_callback = ModelCheckpoint(
    save_top_k = 2,
    monitor="training_loss",
    mode="min",
    filename="RWKV-best-{epoch:02d}-{training_loss:.2f}",
)
reg_checkpoint_callback = ModelCheckpoint(
    every_n_epochs = 100,
    filename="RWKV-reg-{epoch:02d}-{training_loss:.2f}",
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'A simple Implementation of RWKV v4')
    parser.add_argument('--train', action = 'store_true', default = False, help = 'Initialize weights (prior than --weights option)')
    parser.add_argument('-w', '--weights', type = str, help = 'Specify the path of the weights file')
    parser.add_argument('-d', '--dataset_idx', type = str, help = 'Path to the dataset idx file.')
    parser.add_argument('-t', '--tokenizer', type = str, help = 'Path to the tokenizer_file')

    args = parser.parse_args()

    if args.weights != None:
        if not os.path.isfile(args.weights):
            print(f'[FATAL] Weights file {args.weights} doesn\'t exists. Immediately exit with code 1.')
            sys.exit(1)
    
    dataset = BinIdxDataset(args.dataset_idx, context_len, miniepoch_size)

    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    set_seed(20240728)

    cuda_device = torch.device('cuda')

    if args.train:

        model = RWKV_v4(init_weights = True, vocab_size = vocab_size, total_layers = total_layers, emb_size = emb_size,
                        time_mixing_weight_std = time_mixing_weight_std, time_mixing_hidden_size = time_mixing_hidden_size,
                        channel_mixing_weight_std = channel_mixing_weight_std, channel_mixing_hidden_size = channel_mixing_hidden_size, embedding_init_value = 1e-4,
                        adam_betas = adam_betas, learning_rate = learning_rate).to(cuda_device)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, num_workers = 8)
    
        trainer = L.Trainer(devices = gpu_count, max_epochs = epoch_count, log_every_n_steps = 1, callbacks = [best_checkpoint_callback, reg_checkpoint_callback])
        trainer.fit(model = model, train_dataloaders = dataset_loader)

    elif args.weights:

        model = RWKV_v4.load_from_checkpoint(args.weights)

        model.to(cuda_device)

        model.eval()

        tokenizer = PreTrainedTokenizerFast(tokenizer_file = args.tokenizer)
        
        model_input = dataset.get_one_passage()[:96]
        model_input = model_input.view(1, -1)

        count = len(model_input)

        context_len = 448

        init = tokenizer.decode(list(model_input.flatten()))

        while count < context_len:
            model_output = (model(model_input.cuda())[0, -1]).detach().flatten()
            output_token = int(torch.argmax(model_output))
            model_input = torch.concat((model_input, torch.tensor([[output_token, ]])), 1)
            count += 1
            print(f"{count}th iteration")
            if output_token == 0:
                break
        
        print(init)
        print(tokenizer.decode(list(model_input.flatten())))
