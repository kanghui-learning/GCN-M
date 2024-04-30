import pandas as pd
import numpy as np

raw_data_path = '/root/projects/GCN-M/Datasets/METR_LA/metr_la.h5'
val_data_path = '/root/projects/GCN-M/Datasets/METR_LA/missRatio_20.00%_dateTime_val.npz'
train_data_path = '/root/projects/GCN-M/Datasets/METR_LA/missRatio_20.00%_dateTime_train.npz'
test_data_path = '/root/projects/GCN-M/Datasets/METR_LA/missRatio_20.00%_dateTime_test.npz'



raw_df = pd.read_hdf(raw_data_path)
val_data = np.load(val_data_path)
train_data = np.load(train_data_path)
test_data = np.load(test_data_path)


import pdb; pdb.set_trace()