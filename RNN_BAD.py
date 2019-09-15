
import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import tensorflow as tf
import matplotlib.pyplot as plt

import db
import utils

# split data in 80%/10%/10% train/validation/test sets
valid_set_size_percentage = 10 
test_set_size_percentage = 10 

#display parent directory and working directory
print(os.path.dirname(os.getcwd())+':', os.listdir(os.path.dirname(os.getcwd())));
print(os.getcwd()+':', os.listdir(os.getcwd()));


""" reading in the data """
# sym_array = ['/ES', '/CL', '/GC']
sym_array = ['/CL']
main_df = pd.DataFrame()
""" request data from DB """
for sym in sym_array:
  data = db.get_symbol(sym, 1000)
  # print(data)
  sym = sym[1:]
  df = pd.DataFrame.from_dict(data).rename(columns={"open":f"{sym}_open", "high":f"{sym}_high", "low":f"{sym}_low", "close":f"{sym}_close", "volume":f"{sym}_volume"})

  if(len(main_df) == 0):
    main_df = df
  else:
    main_df = main_df.join(df)

print(main_df.head())


# utils.plot_data('CL', main_df)









# function for min-max normalization of stock
def normalize_data(df, sym):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    df[f'{sym}_open'] = min_max_scaler.fit_transform(df[f'{sym}_open'].values.reshape(-1,1))
    df[f'{sym}_high'] = min_max_scaler.fit_transform(df[f'{sym}_high'].values.reshape(-1,1))
    df[f'{sym}_low'] = min_max_scaler.fit_transform(df[f'{sym}_low'].values.reshape(-1,1))
    df[f'{sym}_close'] = min_max_scaler.fit_transform(df[f'{sym}_close'].values.reshape(-1,1))
    return df

# function to create train, validation, test data given stock data and sequence length
def load_data(stock, seq_len):
    data_raw = stock.as_matrix() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - seq_len): 
        data.append(data_raw[index: index + seq_len])
    
    data = np.array(data);
    valid_set_size = int(np.round(valid_set_size_percentage/100*data.shape[0]));  
    test_set_size = int(np.round(test_set_size_percentage/100*data.shape[0]));
    train_set_size = data.shape[0] - (valid_set_size + test_set_size);
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_valid = data[train_set_size:train_set_size+valid_set_size,:-1,:]
    y_valid = data[train_set_size:train_set_size+valid_set_size,-1,:]
    
    x_test = data[train_set_size+valid_set_size:,:-1,:]
    y_test = data[train_set_size+valid_set_size:,-1,:]
    
    return [x_train, y_train, x_valid, y_valid, x_test, y_test]

# choose one stock
df_stock = main_df.copy()
# df_stock.drop(['symbol'],1,inplace=True)
df_stock.drop(['CL_volume'],1,inplace=True)

cols = list(df_stock.columns.values)
print('df_stock.columns.values = ', cols)

# normalize stock
df_stock_norm = df_stock.copy()
df_stock_norm = normalize_data(df_stock_norm, 'CL')
print(df_stock_norm.head())

# create train, test data
seq_len = 20 # choose sequence length
x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(df_stock_norm, seq_len)
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_valid.shape = ',x_valid.shape)
print('y_valid.shape = ', y_valid.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ',y_test.shape)

# utils.plot_norm_data(df_stock_norm, 'CL')


utils.model_validate(x_train,x_test, x_valid, y_train, y_test, y_valid, seq_len)