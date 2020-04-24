import argparse
from configuration import ConfigManager
from data import DatasetManager
from evaluation import evaluate
import json
import logging
import models
import nets
import numpy as np
import os
from sampler import DataSampler
import sys
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s]  %(message)s",
                    datefmt='%H:%M:%S-%d%m%y')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders')
parser.add_argument('--seed', type=int, default=98765,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--data_conf', type=str, default='config/config_data_ml20m.json',
                    help='path to the configuration file for reading the data')
parser.add_argument('--model_conf', type=str, default='config/config_vae.json',
                    help='path to the configuration file for reading the data')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        logger.warning("You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

logger.info("Parametrs: " + str(vars(args)))

ConfigManager(args.data_conf, args.model_conf)
vae_config = ConfigManager.get().model_config
data_config = ConfigManager.get().data_config

logger.info("Data configuration: " + str(data_config))
logger.info("Model configuration: " + str(vae_config))

###############################################################################
# Load data
###############################################################################
data_man = DatasetManager(data_config)
tr_loader = DataSampler(*data_man.training_set, **vae_config.sampler)
val_loader = DataSampler(*data_man.validation_set, **vae_config.sampler, shuffle=False)
te_loader = DataSampler(*data_man.test_set, **vae_config.sampler, shuffle=False)

###############################################################################
# Training the model
###############################################################################
dec_dims = [200, 600, data_man.n_items]
model = nets.MultiVAE_net(dec_dims).to(device)
vae = models.MultiVAE(model, **vae_config.model)
logger.info("Model: " + str(vae))
vae.train(tr_loader, val_loader, **vae_config.train)

###############################################################################
# Test the model
###############################################################################
stats = evaluate(vae, te_loader, **vae_config.test)
str_stats = " | ".join([f"{k} {np.mean(v):.5f} ({np.std(v)/np.sqrt(len(v)):.4f})" for k,v in stats.items()])
logger.info(f'| final evaluation | {str_stats} |')
