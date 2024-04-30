import argparse, configparser
import os
import torch
import wandb
import pytz
import datetime
from exp.exp_GCNM import Exp_GCNM
from utils.tools import init_seeds

parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configs/PEMS_BAY_GCNM.conf', type=str,
                    help="configuration file path")
parser.add_argument("--itr", default=1, type=int,
                    help="the iteration round for the model")
parser.add_argument("--debug", default=False, type=bool,
                    help="if debug")

args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
if 'DEFAULT' not in config:
    config['DEFAULT'] = {}
config['DEFAULT']['debug'] = str(args.debug)  # Store booleans as string
config_dict = {section: dict(config.items(section)) for section in config.sections()}

if not args.debug:
    timezone = pytz.timezone('Asia/Shanghai')
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_utc8 = now_utc.astimezone(timezone)
    formatted_time = now_utc8.strftime("%Y%m%d%H%M%S")
    
    name = 'GCNM-' + config['Data']['dataset_name'] + '-Mask' + "{:.2f}".format(1 - float(config['Data']['mask_ones_proportion']))\
        + '-L' + str(config['Model']['L']) + '-nd' + str(config['Model']['nd']) + '-nw' + str(config['Model']['nw'])\
        +'-seed' + str(config['Model']['seed']) + '-' + formatted_time
    wandb.init(project='Missing Value Forecasting', name=name)
    wandb.config.update(config_dict)
    wandb.config["iteration"] = args.itr

init_seeds(int(config['Model']['seed']))

for ii in range(args.itr):
    print('Main Interation Round {}'.format(ii))

    exp = Exp_GCNM(config) # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(config))
    exp.train()

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(config))
    exp.test()

    torch.cuda.empty_cache()
