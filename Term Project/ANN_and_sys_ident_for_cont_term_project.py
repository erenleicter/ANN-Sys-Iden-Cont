import tensorflow as tf
import pandas as pd
import numpy as np
from math import pi
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops

# Import the monthly milk production data, Index data = Month
aerobeam_data = pd.read_csv("log_final.csv")

# Train Test Split
aerobeam_data.columns = ['time', 'pitch', 'u']
aerobeam_data.pitch = aerobeam_data.pitch*pi/180

train_set_X, test_set_X, train_set_Y, test_set_Y = train_test_split(aerobeam_data['u'], aerobeam_data['pitch'],
                                                                    test_size=0.1, shuffle=False)

tf.reset_default_graph()


def custom_loss_function(y_desired, u_pred):
    loss = K.mean(math_ops.square(u_pred - y_desired), axis=-1)
    return loss


def data_sequences(input_data, output_data, input_size, output_size):
    length = len(input_data)
    dimension = length - input_size + 1

    input_sequence = np.zeros([dimension, input_size + output_size])
    for x in range(dimension):
        for y in range(input_size):
            input_sequence[x][y] = input_data.iloc[x + y]
            # input_sequence[x][y] = input_data[x + y]

    for x in range(dimension):
        for y in range(input_size, input_size + output_size):
            input_sequence[x][y] = output_data.iloc[x + y - input_size]
            # input_sequence[x][y + input_size] = output_data[x + y]

    output_sequence = np.zeros([dimension, 1])
    for x in range(dimension):
        output_sequence[x] = output_data.iloc[x + input_size - 1]
        # output_sequence[x] = output_data[x + input_size - 1]

    return input_sequence, output_sequence

# The Constants

num_inputs = 4
# num_time_steps = 120
num_neurons_per_layer = 8
num_outputs = 3
learning_rate = 0.025
num_iterations = 500

train_set_X, train_set_Y = data_sequences(train_set_X, train_set_Y, num_inputs, num_outputs)
test_set_X, test_set_Y = data_sequences(test_set_X, test_set_Y, num_inputs, num_outputs)

print(train_set_X[3])
print(train_set_Y[3])
# Select the model mode
nn_keras_model = models.Sequential()

# Add input and hidden layers
nn_keras_model.add(layers.Dense(units=num_neurons_per_layer, input_dim=train_set_X.shape[1], activation='relu'))
nn_keras_model.add(layers.Dense(units=num_neurons_per_layer, activation='relu'))
nn_keras_model.add(layers.Dense(units=num_neurons_per_layer, activation='relu'))
nn_keras_model.add(layers.Dense(units=num_neurons_per_layer, activation='relu'))

# Output Layer
nn_keras_model.add(layers.Dense(units=1))

# Compile the model
nn_keras_model.compile(optimizer='rmsprop',
                       # loss='mse',
                       loss=custom_loss_function,
                       metrics=['mse'])

print("****")
nn_keras_model.summary()
print("****")

# Training
nn_keras_model.fit(train_set_X, train_set_Y, epochs=num_iterations)

# Predictions
metric = nn_keras_model.evaluate(train_set_X, train_set_Y)
print(nn_keras_model.metrics_names)
print(metric)

pred = nn_keras_model.predict(test_set_X)
print("Prediction shape: {}").format(pred.shape)
print("Test Y shape: {}").format(test_set_Y.shape)

time = np.linspace(10, 10*test_set_Y.shape[0], test_set_Y.shape[0])

fig, ax = plt.subplots()
ax.plot(time, pred*180/pi, label='Predictions', lw=2, ls='--')
ax.plot(time, test_set_Y*180/pi, label='Real Output')
ax.set_title('Comparison of the predictions and real output of the Aerobeam system ', fontsize='x-large')
ax.set_xlabel('Time (ms)', fontsize='large')
ax.set_ylabel('Degree', fontsize='large')
ax.legend(loc='upper left', shadow=True, fontsize='x-large')

ax.text(0.94, 0.98, 'MSE: {:.5}'.format(metric[1]*180*180/(pi*pi)), horizontalalignment='center', verticalalignment='center',
        bbox=dict(facecolor='red', alpha=0.5), fontsize='x-large', transform=ax.transAxes)
ax.grid(True)

plt.show()

nn_keras_model.save('ANN_Term_Project.h5')
# with tf.Session() as sess:
#     writer = tf.summary.FileWriter("./ANN_Term_Project.h5", sess.graph)

# print "Control history :\n {}".format(control_history)



