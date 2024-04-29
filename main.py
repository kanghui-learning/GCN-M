import argparse, configparser
import os
import torch
import wandb
from exp.exp_GCNM import Exp_GCNM

parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configs/PEMS_BAY_GCNM.conf', type=str,
                    help="configuration file path")
parser.add_argument("--itr", default=1, type=int,
                    help="the iteration round for the model")

args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
config_dict = {section: dict(config.items(section)) for section in config.sections()}

wandb.init(project='Missing Value Forecasting', name='GCNM-METR-20%-24')
wandb.config.update(config_dict)
wandb.config["iteration"] = args.itr

for ii in range(args.itr):
    print('Main Interation Round {}'.format(ii))

    exp = Exp_GCNM(config) # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(config))
    exp.train()

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(config))
    exp.test()

    torch.cuda.empty_cache()
