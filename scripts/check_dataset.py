import pdb
import pandas as pd
import numpy as np
import sys
sys.path.append('/root/projects/GCN-M/')
from data.gcnm_utils import DataLoader

raw_data_path = '/root/projects/GCN-M/Datasets/METR_LA/metr_la.h5'
val_data_path = '/root/projects/GCN-M/Datasets/METR_LA/missRatio_20.00%_dateTime_val.npz'
train_data_path = '/root/projects/GCN-M/Datasets/METR_LA/missRatio_20.00%_dateTime_train.npz'
test_data_path = '/root/projects/GCN-M/Datasets/METR_LA/missRatio_20.00%_dateTime_test.npz'

batch_size = 32

raw_df = pd.read_hdf(raw_data_path)
val_data = np.load(val_data_path)
train_data = np.load(train_data_path)
test_data = np.load(test_data_path)

x_train = train_data['x']
date_train = train_data['dateTime']
y_train = train_data['y']

x_val = val_data['x']
date_val = val_data['dateTime']
y_val = val_data['y']

x_test = test_data['x']
date_test = test_data['dateTime']
y_test = test_data['y']

train_loader = DataLoader(x_train, date_train, y_train, batch_size)
val_loader = DataLoader(x_val, date_val, y_val, batch_size)
test_loader = DataLoader(x_test, date_test, y_test, batch_size)

pdb.set_trace()

for i, (batch_x, batch_dateTime, batch_y) in enumerate(val_loader.get_iterator()):
    if np.count_nonzero(batch_y) == 0:
        pdb.set_trace()


pdb.set_trace()