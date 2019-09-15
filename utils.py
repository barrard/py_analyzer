import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf


def plot_data(sym, main_df):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(main_df[f'{sym}_close'].values, color='red', label='open')

    plt.title('stock price')
    plt.xlabel('time [days]')
    plt.ylabel('price')
    plt.legend(loc='best')
    # plt.show()

    plt.subplot(1, 2, 2)
    plt.plot(main_df[f'{sym}_volume'].values, color='black', label='volume')
    plt.title('stock volume')
    plt.xlabel('time [days]')
    plt.ylabel('volume')
    plt.legend(loc='best')

    plt.show()


def plot_norm_data(df_stock_norm, sym):
    plt.figure(figsize=(15, 5))
    plt.plot(df_stock_norm[f'{sym}_open'].values, color='red', label='open')
    plt.plot(df_stock_norm[f'{sym}_close'].values, color='green', label='low')
    plt.plot(df_stock_norm[f'{sym}_low'].values, color='blue', label='low')
    plt.plot(df_stock_norm[f'{sym}_high'].values, color='black', label='high')
    #plt.plot(df_stock_norm.volume.values, color='gray', label='volume')
    plt.title('stock')
    plt.xlabel('time [days]')
    plt.ylabel('normalized price/volume')
    plt.legend(loc='best')
    plt.show()


def model_validate(x_train, x_test, x_valid, y_train, y_test, y_valid, seq_len):
    # Basic Cell RNN in tensorflow

    index_in_epoch = 0
    perm_array = np.arange(x_train.shape[0])
    np.random.shuffle(perm_array)

    # function to get the next batch
    def get_next_batch(batch_size, index_in_epoch, x_train, perm_array):
        # global index_in_epoch, x_train, perm_array
        start = index_in_epoch
        index_in_epoch += batch_size

        if index_in_epoch > x_train.shape[0]:
            np.random.shuffle(perm_array)  # shuffle permutation array
            start = 0  # start next epoch
            index_in_epoch = batch_size

        end = index_in_epoch
        return x_train[perm_array[start:end]], y_train[perm_array[start:end]]

    # parameters
    n_steps = seq_len-1
    n_inputs = 4
    n_neurons = 200
    n_outputs = 4
    n_layers = 2
    learning_rate = 0.001
    batch_size = 50
    n_epochs = 10#100
    train_set_size = x_train.shape[0]
    test_set_size = x_test.shape[0]

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_outputs])

    # use Basic RNN Cell
    layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.elu)
              for layer in range(n_layers)]

    # use Basic LSTM Cell
    # layers = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.nn.elu)
    #          for layer in range(n_layers)]

    # use LSTM Cell with peephole connections
    # layers = [tf.contrib.rnn.LSTMCell(num_units=n_neurons,
    #                                  activation=tf.nn.leaky_relu, use_peepholes = True)
    #          for layer in range(n_layers)]

    # use GRU cell
    # layers = [tf.contrib.rnn.GRUCell(num_units=n_neurons, activation=tf.nn.leaky_relu)
    #          for layer in range(n_layers)]

    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
    rnn_outputs, states = tf.nn.dynamic_rnn(
        multi_layer_cell, X, dtype=tf.float32)

    stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
    stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
    outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
    outputs = outputs[:, n_steps-1, :]  # keep only last output of sequence

    # loss function = mean squared error
    loss = tf.reduce_mean(tf.square(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    # run graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for iteration in range(int(n_epochs*train_set_size/batch_size)):
            # fetch the next training batch
            x_batch, y_batch = get_next_batch(batch_size, index_in_epoch, x_train, perm_array)
            sess.run(training_op, feed_dict={X: x_batch, y: y_batch})
            if iteration % int(5*train_set_size/batch_size) == 0:
                mse_train = loss.eval(feed_dict={X: x_train, y: y_train})
                mse_valid = loss.eval(feed_dict={X: x_valid, y: y_valid})
                print('%.2f epochs: MSE train/valid = %.6f/%.6f' % (
                    iteration*batch_size/train_set_size, mse_train, mse_valid))

        y_train_pred = sess.run(outputs, feed_dict={X: x_train})
        y_valid_pred = sess.run(outputs, feed_dict={X: x_valid})
        y_test_pred = sess.run(outputs, feed_dict={X: x_test})

    print(y_train.shape)
    show_predictions(y_train, y_valid, y_test, y_train_pred, y_valid_pred, y_test_pred)


def show_predictions(y_train, y_valid, y_test, y_train_pred, y_valid_pred, y_test_pred):
  ft = 0 # 0 = open, 1 = close, 2 = highest, 3 = lowest

  ## show predictions
  plt.figure(figsize=(15, 5))
  plt.subplot(1,2,1)

  plt.plot(np.arange(y_train.shape[0]), y_train[:,ft], color='blue', label='train target')

  plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_valid.shape[0]), y_valid[:,ft],
          color='gray', label='valid target')

  plt.plot(np.arange(y_train.shape[0]+y_valid.shape[0],
                    y_train.shape[0]+y_test.shape[0]+y_test.shape[0]),
          y_test[:,ft], color='black', label='test target')

  plt.plot(np.arange(y_train_pred.shape[0]),y_train_pred[:,ft], color='red',
          label='train prediction')

  plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0]+y_valid_pred.shape[0]),
          y_valid_pred[:,ft], color='orange', label='valid prediction')

  plt.plot(np.arange(y_train_pred.shape[0]+y_valid_pred.shape[0],
                    y_train_pred.shape[0]+y_valid_pred.shape[0]+y_test_pred.shape[0]),
          y_test_pred[:,ft], color='green', label='test prediction')

  plt.title('past and future stock prices')
  plt.xlabel('time [days]')
  plt.ylabel('normalized price')
  plt.legend(loc='best')

  plt.subplot(1,2,2)

  plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_test.shape[0]),
          y_test[:,ft], color='black', label='test target')

  plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0]+y_test_pred.shape[0]),
          y_test_pred[:,ft], color='green', label='test prediction')

  plt.title('future stock prices')
  plt.xlabel('time [days]')
  plt.ylabel('normalized price')
  plt.legend(loc='best');

  corr_price_development_train = np.sum(np.equal(np.sign(y_train[:,1]-y_train[:,0]),
              np.sign(y_train_pred[:,1]-y_train_pred[:,0])).astype(int)) / y_train.shape[0]
  corr_price_development_valid = np.sum(np.equal(np.sign(y_valid[:,1]-y_valid[:,0]),
              np.sign(y_valid_pred[:,1]-y_valid_pred[:,0])).astype(int)) / y_valid.shape[0]
  corr_price_development_test = np.sum(np.equal(np.sign(y_test[:,1]-y_test[:,0]),
              np.sign(y_test_pred[:,1]-y_test_pred[:,0])).astype(int)) / y_test.shape[0]

  print('correct sign prediction for close - open price for train/valid/test: %.2f/%.2f/%.2f'%(
      corr_price_development_train, corr_price_development_valid, corr_price_development_test))
  plt.show()