# univariate stacked lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import pickle

import db
import utils


# split a univariate sequence



# define input sequence
# raw_seq = db.get_symbol('/ES', 10000)

# choose a number of time steps
n_steps = 40
n_features = 1
# split into samples
# X, y = utils.split_sequence(raw_seq, n_steps)
# # reshape from [samples, timesteps] into [samples, timesteps, features]
# print(X.shape)
# print(X)
# X = X.reshape((X.shape[0], X.shape[1], n_features))
# print(X)
# print(X.shape)
# define model
# model = Sequential()
# model.add(LSTM(50, activation='relu', return_sequences=True,
#                input_shape=(n_steps, n_features)))
# model.add(LSTM(50, activation='relu'))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')
# fit model
filename = 'stacked_LSTM_model.sav'

# model.fit(X, y, epochs=15, verbose=1)
model = pickle.load(open(filename, 'rb'))

# pickle.dump(model, open(filename, 'wb'))


# demonstrate prediction
test_x = []
test_data  = db.get_symbol('/ES', n_steps)
for i in range(len(test_data)):
  tmp = [test_data[i]['close']]
  test_x.append(tmp)
test_x = array(test_x)
x_input = test_x.reshape((1, n_steps, 1))
print(x_input)
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=1)
print(f'Time stanp start {test_data[0]["start_timestamp"]} ${test_data[0]["close"]}')
print(f'Time stanp end {test_data[len(test_data)-1]["start_timestamp"]} ${test_data[len(test_data)-1]["close"]}')
print(yhat)
