import argparse
from configuration import ConfigurationManager
import data
import json
import logging
import models
import nets
import os
import sys
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.DEBUG,
                    format="[%(asctime)s]  %(message)s",
                    datefmt='%H:%M:%S-%d%m%y',
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders')
parser.add_argument('--total_anneal_steps', type=int, default=200000,
                    help='the total number of gradient updates for annealing')
parser.add_argument('--anneal_cap', type=float, default=0.2,
                    help='largest annealing parameter')
parser.add_argument('--seed', type=int, default=98765,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
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
#with open(args.model_conf, 'r') as f:
#    vae_config = json.load(f)

logger.info("Parametrs: " + str(vars(args)))

ConfigurationManager(args.data_conf, args.model_conf)
vae_config = ConfigurationManager.get_instance().model_config
data_config = ConfigurationManager.get_instance().data_config

logger.info("Data configuration: " + str(data_config))
logger.info("Model configuration: " + str(vae_config))

###############################################################################
# Load data
###############################################################################
batch_size = vae_config.batch_size
data_manager = data.DatasetManager(data_config)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
tr_loader = DataLoader(data_manager.training_set, batch_size=batch_size, shuffle=True, **kwargs)
val_loader = DataLoader(data_manager.validation_set, batch_size=batch_size, shuffle=False, **kwargs)
te_loader = DataLoader(data_manager.test_set, batch_size=batch_size, shuffle=False, **kwargs)

###############################################################################
# Training the model
###############################################################################
dec_dims = [200, 600, data_manager.n_items]
model = nets.MultiVAE_net(dec_dims).to(device)
logger.info("Network: " + str(model))
#vae = models.MultiVAE(model, num_epochs=vae_config.num_epochs, learning_rate=vae_config.learning_rate)
vae = models.MultiVAE(model,
                      beta=args.anneal_cap,
                      anneal_steps=args.total_anneal_steps,
                      num_epochs=vae_config.num_epochs,
                      learning_rate=vae_config.learning_rate)
logger.info("Model: " + str(vae))
vae.train(tr_loader, val_loader, vae_config.valid_metrics[0], vae_config.verbose)

###############################################################################
# Test the model
###############################################################################
#test_loss, stats = vae.evaluate(te_loader, vae_config["test_metrics"])

#for metric in vae_config["test_metrics"]
#str_stats = " | ".join([f"{k} {v:.3f}" for k,v in stats.items()])
#logger.info('| final evaluation | test loss {test_loss:.2f} | {str_stats} |')
