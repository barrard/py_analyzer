from numpy import array

# univariate lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from pprint import pprint

# split a univariate sequence into samples


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
  
        seq_x, seq_y = process_ohlc(seq_x, seq_y)

        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def process_ohlc(seq_x, seq_y):
  clean_x=[]
  clean_y=[seq_y['close']]
  for i in range(len(seq_x)):
    each_x = [seq_x[i]['close']]
    clean_x.append(each_x)

  return array(clean_x), array(clean_y)



def _process_ohlc(seq_x, seq_y):
  clean_x=[]
  clean_y=[seq_y['open'], seq_y['high'], seq_y['low'], seq_y['close'], seq_y['volume']]
  for i in range(len(seq_x)):
    each_x = [seq_x[i]['open'], seq_x[i]['high'], seq_x[i]['low'], seq_x[i]['close'], seq_x[i]['volume']]
    clean_x.append(each_x)

  return array(clean_x), array(clean_y)


""" 
            ----------Vanilla LSTM
 A Vanilla LSTM is an LSTM model that has a single hidden 
layer of LSTM units, and an output layer used to make a prediction.

 """

# define model
""" We are working with a univariate series, so the number of features is one, for one variable """


def uni_variant_LSTM(X, y, n_steps, n_features):
    model = Sequential()
    """ The number of time steps as input is the number we chose when preparing our dataset as an argument to the split_sequence() function. """
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # fit model
    model.fit(X, y, epochs=20, verbose=1)
    return model


def predict(model, x_input, n_steps, n_features):

    yhat = model.predict(x_input, verbose=0)
    print(yhat)


def test(): print('test')
