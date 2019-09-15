import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from collections import deque
import random
import db
import time

import tensorflow as tf
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import  Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import  TensorBoard, ModelCheckpoint

SEQ_LEN = 60
FUTURE_PREIOD_PREDICT = 3

COMM_TO_PREDICT = 'ES'
EPOCHS = 10
BATCH_SIZE = 64
NAME= f"{SEQ_LEN}-SEQ-{COMM_TO_PREDICT}--PRED-{int(time.time())}"

def preprocess_df(df):
  df = df.drop('future', 1)

  for col in df.columns:
    if col != "target":
      """ target column does not need to be preprocessed """
      df[col] = df[col].pct_change() #normalizing the data
      df.dropna(inplace=True)
      df[col] = preprocessing.scale(df[col].values)

  df.dropna(inplace=True)#just to make sure...
  sequential_data = []
  prev_days = deque(maxlen=SEQ_LEN)
  # print(df.head(20))

  for i in df.values:
    prev_days.append([n for n in i[:-1]])#last columns is target, so we drop it
    if(len(prev_days) == SEQ_LEN):
      # print('Prev days')
      # print(prev_days)
      # print(i[-1])
      sequential_data.append([np.array(prev_days), i[-1]])
  random.shuffle(sequential_data)
  buys = []
  sells = []

  for seq, target in sequential_data:
    if target ==0:
      sells.append([seq, target])
    elif target ==1:
      buys.append([seq, target])

  random.shuffle(buys)
  random.shuffle(sells)

  lower = min(len(buys), len(sells))
  # print(lower)

  buys = buys[:lower]
  sells = sells[:lower]

  sequential_data = buys+sells
  random.shuffle(sequential_data)

  X = []
  y = []

  for seq, target in sequential_data:
    X.append(seq)
    y.append(target)
  # print(X)
  return(np.array(X), y)


def classify(current, future):
  if float(future) > float(current):
    return 1
  else:
    return 0

sym_array = ['/ES', '/CL', '/GC']
main_df = pd.DataFrame()
""" request data from DB """
for sym in sym_array:
  data = db.get_symbol(sym, 10000)
  sym = sym[1:]
  df = pd.DataFrame.from_dict(data).rename(columns={"close":f"{sym}_close", "volume":f"{sym}_volume"})

  if(len(main_df) == 0):
    main_df = df
  else:
    main_df = main_df.join(df)



main_df['future'] = main_df[f"{COMM_TO_PREDICT}_close"].shift(-FUTURE_PREIOD_PREDICT)

main_df['target'] = list(map(classify, main_df[f"{COMM_TO_PREDICT}_close"], main_df['future']))

# print(main_df.head())

indx = sorted(main_df.index.values)
last_5pct = indx[-int(0.05*len(indx))]
# print(len(indx))
# print(last_5pct)


validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

# preprocess_df(main_df)
train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)


print(f"train data: {len(train_x)} validataion:{len(validation_x)}")
print(f"dont buys: {train_y.count(0)} buys:{train_y.count(1)}")
print(f"VALIDATION dont buys: {validation_y.count(0)} buys:{validation_y.count(1)}")

model = Sequential()


model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization)

# model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
# model.add(Dropout(0.1))#maybe 0.2??
# model.add(BatchNormalization)

# model.add(LSTM(128, input_shape=(train_x.shape[1:])))
# model.add(Dropout(0.2))
# model.add(BatchNormalization)

# model.add(Dense(32, activation = "relu"))#TNH activation? which is what CuDNN uses

# model.add(Dropout(0.2))

# model.add(Dense(2, activation="softmax"))

# opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# model.compile(loss='sparse_categorical_crossentropy',
#       optimizer=opt,
#       metrics=['accuracy'])

# tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

# filepath= "RNN_Final-{epoch:02d}-{val_acc:.3f}"
# checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'))