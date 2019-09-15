from numpy import array

# univariate lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from pprint import pprint

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
