"""
MIT License

Copyright (c) 2019 erenleicter

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
from matplotlib import pyplot as plt

# Input size
input_size = 10
output_size = 10

# Iteration number
iterations = 50000

#Learning Rate
lr = 0.001

# Hidden Layer Sizes
hidden_size = [10, 12, 15, 12, 10]
# hidden_size = [5, 8, 10, 15, 20]


f = 50
end_time = 5

time = np.linspace(0, 1*end_time, f*end_time)
numpy_sin = np.linspace(0, 1*end_time, f*end_time)
numpy_sin = np.sin(2*np.pi*numpy_sin)

X_validation = np.ones((input_size, f*end_time - input_size - output_size + 1))
T_validation = np.ones((output_size, f*end_time - input_size - output_size + 1))

for y in range(0, f*end_time - input_size - output_size + 1):
    for x in range(0, input_size):
        X_validation[x][y] = numpy_sin[y + x]

    for x in range(0, output_size):
        T_validation[x][y] = numpy_sin[y + x + input_size]

X_validation = X_validation.T
T_validation = T_validation.T

noise = np.random.randn(f*end_time - input_size - output_size + 1, input_size)*0.01
X = np.copy(X_validation) + noise

T = T_validation

X_desired = X_validation.T
# X: test input
X_test = np.ones(input_size)
X_test = np.copy(X_validation[0][0:input_size]) + 10*noise[0][0:input_size]


X_test = X_test.T


class ArtificialNeuralNetwork(object):
    """
    This class initializes n input m output one node neuron. Hence no hidden layers included.
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate, activation_type="sigmoid", batch_divide=5):
        """
        For n x m node:

        :param input_size: n (int).
        :param hidden_size: [h1, h2, ... , hn] (list).
        :param output_size: m (int).
        :param learning_rate: Learning rate (float).
        """
        self.inputSize = input_size
        self.outputSize = output_size
        self.hiddenLayer = len(hidden_size)
        self.hiddenSize = hidden_size
        self.lr = learning_rate
        self.activation_type = activation_type
        self.batch_divide = batch_divide

        self.error = 0
        self.y_list = []
        self.V_list = []
        self.delta_list = []


        weight_list = []

        if not self.hiddenLayer:
            weight = np.random.randn(self.inputSize, self.outputSize)
            weight_list.append(weight)
            self.Weights = weight_list

        else:
            weight = np.random.randn(self.inputSize, self.hiddenSize[0])
            weight_list.append(weight)

            for element in range(1, self.hiddenLayer):
                weight = np.random.randn(self.hiddenSize[element - 1], self.hiddenSize[element])

                weight_list.append(weight)

            weight = np.random.randn(self.hiddenSize[-1], self.outputSize)
            weight_list.append(weight)

            # self.Weights = np.array((weight_list), dtype=np.float64)
            self.Weights = weight_list

    @staticmethod
    def activation_function(v, type="sigmoid"):
        """
        This staticmethod only contains sigmoid function as the activation function.

        :param v: V net input.
        :return: Output value of the node.
        """

        # Sigmoid function.
        if type == "sigmoid":
            return 1 / (1 + np.exp(-v))

        elif type == "tanh":
            return np.tanh(v)

    def feed_forward(self, x):
        """
        This method returns the output for this one node network.

        :param x: Input data, numpy.array(n, x).
        :return: The output.
        """
        self.y_list = []
        self.V_list = []

        V = np.dot(x, self.Weights[0])
        self.V_list.append(V)
        y = self.activation_function(V, self.activation_type)
        self.y_list.append(y)

        if not self.hiddenLayer:
            return y

        else:

            for element in range(1, self.hiddenLayer):
                V = np.dot(self.y_list[element - 1], self.Weights[element])
                self.V_list.append(V)
                y = self.activation_function(V, self.activation_type)
                self.y_list.append(y)

            V = np.dot(self.y_list[-1], self.Weights[-1])
            self.V_list.append(V)
            y = self.activation_function(V, self.activation_type)
            self.y_list.append(y)
            return y

    def activation_function_derivative(self, v, type="sigmoid"):
        """
        This method only contains derivative of sigmoid function as the activation function.

        :param v: V net input.
        :return: Derivative of F(V).
        """
        # Derivative of sigmoid.
        if type == "sigmoid":
            return self.activation_function(v) * (1 - self.activation_function(v))

        elif type == "tanh":
            return 1 - (self.activation_function(v, type="tanh"))**2

    def back_propagation(self, x, t, y):
        """
        Back propagation algorithm.

        :param x: Input data, numpy.array(n, x).
        :param t: Desired output data, numpy.array(n).
        :param y: Output value of the node.
        :return: None.
        """
        self.delta_list = []
        self.error = t - y
        delta = self.error * (self.activation_function_derivative(self.V_list[-1], self.activation_type))
        self.delta_list.insert(0, delta)

        if not self.hiddenLayer:
            self.Weights[-1] += self.lr * x.T.dot(self.delta_list[-1])

        else:
            for element in range(1, self.hiddenLayer + 1):
                delta = (self.delta_list[-element].dot(self.Weights[-element].T)) *\
                        self.activation_function_derivative(self.V_list[-(element + 1)], self.activation_type)

                self.delta_list.insert(0, delta)

            self.weight_update(x)

    def weight_update(self, x):
        self.Weights[0] += self.lr * x.T.dot(self.delta_list[0])

        for element in range(1, self.hiddenLayer + 1):
            self.Weights[element] += self.lr * self.y_list[element-1].T.dot(self.delta_list[element])

    def train(self, input, desired, iterate, outputs_lr, total_error_lr, valid_error_lr, weights_lr):
        """
        This method does feed forward and back propagation in each iteration.

        :param x: Input data, numpy.array(n, x).
        :param t: Desired output data, numpy.array(n).
        :param iterate: Number of iteration.
        :param outputs_lr: Outputs for test data in each iteration.
        :param total_error_lr: Total errors in each iteration.
        :param valid_error_lr: Validation errors in each iteration.
        :param weights_lr: Weights data in each iteration.
        :return: None.
        """

        batch_input = np.array_split(input, self.batch_divide)
        batch_desired = np.array_split(desired, self.batch_divide)

        for batch in range(len(batch_input)):
            x = batch_input[batch]
            t = batch_desired[batch]

            for ith_element in xrange(iterate):
                y = self.feed_forward(x)
                self.back_propagation(x, t, y)

                outputs_lr.append(self.feed_forward(X_test))
                total_error_lr.append(self.total_error())
                valid_error_lr.append(self.validation_total_error())
                for kth_element in range(self.hiddenLayer + 1):
                    weights_lr.append(self.Weights[kth_element].copy())


    @staticmethod
    def save_weights(name, w):
        """
        This static method saves the weights as .txt file.

        :param name: Name of the file, string.
        :param w: Weight vector, np.array().
        :return: None.
        """
        np.savetxt("weights.txt", w, fmt="%s")

    def use_same_weights(self, weights):
        """
        This method allows the using the saved weights.

        :param weights: np.rand(n, m)
        :return:
        """
        self.W1 = weights.copy()

    def test(self, x):
        """
        This method just test the current weighted network.

        :param x: Input data, numpy.array(n, x).
        :return:
        """
        return self.feed_forward(x)

    def validation_total_error(self):
        """
        This method calculates the validation error.

        :return:
        """
        error_validation = T_validation - self.feed_forward(X_validation)
        total_error = error_validation.sum()
        return total_error ** 2 / 2

    def total_error(self):
        """
        This method calculates the validation error.

        :return:
        """
        e = self.error.sum()
        return e ** 2 / 2


# Instance
node = ArtificialNeuralNetwork(input_size, hidden_size, output_size, lr, activation_type="tanh")

total_error_lr = []
valid_error_lr = []
outputs_lr = []
weights_lr = []

node.train(X, T, iterations, outputs_lr, total_error_lr, valid_error_lr, weights_lr)

iterations_fig = np.arange(1, iterations+1)


# axs[0].semilogx(iterations_fig, outputs_lr, lw=4, ls='-.',)  # c='black')

# axs[0].grid(True)

final_results = node.test(X_test)
final_results = np.concatenate((X_test, final_results), axis=None)

print("Final results: \n{}".format(final_results))
print("Validation Total Error results: \n{}".format(node.validation_total_error()))
print("Total Error results: \n{}".format(node.total_error()))
fig, axs = plt.subplots(2, 1)

axs[0].set_title('Total Training error and Validation error vs Iteration (In Logarithmic Scale)')
axs[0].semilogx(total_error_lr, label='Total error')
axs[0].semilogx(valid_error_lr, label='Validation error', ls='--')

axs[1].set_title('Test')
axs[1].plot(X_test, label='Test input')
axs[1].plot(final_results, label='Test output', lw=2, ls='--')
axs[1].plot(X_desired[0][0:20], label='Desired output', lw=4, ls=':')

axs[0].grid(True)
axs[1].grid(True)

axs[0].legend(loc='upper left', shadow=True, fontsize='medium')
axs[1].legend(loc='upper left', shadow=True, fontsize='medium')
plt.show()

# print(node.Weights)
