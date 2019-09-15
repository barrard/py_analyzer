import db
import LSTM_utils
import pprint
import json
from pprint import pprint
from numpy import array
import pickle




""" Get the data for ES, 100 prices """
data = db.get_symbol('/ES', 10000)

""" how many values in a sequence """
n_steps = 60


""" Split the data into sequences """
""" len(X) = 20   len(y) = 1 """
X, y, = LSTM_utils.split_sequence(data, n_steps)

print(X.shape)#(80, 20, 1)
print(y.shape)#(80, 1)



filename = 'univar_LSTM_model.sav'

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
print(X.shape)
univar_LSTM_model = pickle.load(open(filename, 'rb'))
# univar_LSTM_model = LSTM_utils.uni_variant_LSTM(X, y, n_steps, n_features)
pickle.dump(univar_LSTM_model, open(filename, 'wb'))

""" making test data """
test_x = []
test_data = data[-n_steps:]
for i in range(len(test_data)):
  tmp = [data[i]['close']]
  test_x.append(tmp)
test_x = array(test_x)
test_x = test_x.reshape((1, n_steps, 1))

pprint(test_x)
pprint(test_x.shape)
LSTM_utils.predict(univar_LSTM_model, test_x, n_steps, n_features)