# Eric Yeats
# February 2, 2020
# Simple Multi-Layer Perceptron Interface with Backpropagation Algorithm
# Useful for testing modifications to neuron models and/or the backpropagation algorithm
# Thank you Jason Brownlee for the guide "How to Code a Neural Network with Backpropagation In Python (from scratch)"
# It's excellent, available here https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

import numpy as np 


DTYPE = np.float64

class Network:
    """
    MLP data structure that encapsulates input, output, hidden layers, activations, weights, and biases
    includes forward propagation, backward propagation, error calculation, and weight update
    """

    def __init__(self, n_inputs, lrn_rate=0.1):
        self.input = np.zeros((n_inputs), dtype=DTYPE)
        self.expected = None
        self.layers = list()
        self.weights = list()
        self.biases = list()
        self.act_funcs = list()
        self.act_deriv_funcs = list()
        self.errors = list()
        self.lrn_rate = lrn_rate

    def set_input(self, input_data):
        if (type(input_data) != type(self.input)):
            raise ValueError('input data must be numpy ndarray')
        if (self.input.shape != input_data.shape):
            raise ValueError('input data shape must match that of input neurons')
        self.input = input_data

    def set_expected(self, expected_out_data):
        self.expected = expected_out_data

    def get_output(self):
        return self.layers[-1]

    @staticmethod
    def relu(nrn_preact): # activation calculation for single neuron
        act = nrn_preact
        if (act < 0.0):
            act = 0.0
        elif (act > 1.0):
            act = 1.0
        return act

    @staticmethod
    def relu_deriv(nrn_preact):
        act_deriv = 1.0
        if (nrn_preact <= 0.0 or nrn_preact >= 1.0):
            act_deriv = 0.0001
        return act_deriv

    @staticmethod
    def relu_vect(in_arr):
        return np.vectorize(Network.relu, otypes=[DTYPE])(in_arr)

    @staticmethod
    def relu_deriv_vect(in_arr):
        return np.vectorize(Network.relu_deriv, otypes=[DTYPE])(in_arr)

    @staticmethod
    def sigmoid(nrn_preact):
        return (1.0/(1.0+np.exp(-1.0*nrn_preact)))

    @staticmethod
    def sigmoid_deriv(nrn_preact):
        return nrn_preact * (1.0 - nrn_preact)

    @staticmethod
    def sigmoid_vect(in_arr):
        return np.vectorize(Network.sigmoid, otypes=[DTYPE])(in_arr)

    @staticmethod
    def sigmoid_deriv_vect(in_arr):
        return np.vectorize(Network.sigmoid_deriv, otypes=[DTYPE])(in_arr)

    def add_layer(self, n_neurons, act='sigmoid'):
        # add to parallel weight list
        if (len(self.layers) != 0): # not first layer added
            self.weights.append(np.random.rand(self.layers[-1].shape[0], n_neurons)/np.float64(n_neurons))
        else: # the first layer added
            self.weights.append(np.random.rand(self.input.shape[0], n_neurons)/np.float64(n_neurons))
        # add to parallel bias list
        self.biases.append(np.random.rand(n_neurons))
        # add layer state
        self.layers.append(np.zeros((n_neurons,), dtype=DTYPE))
        # keep track of the activation function for the layer
        if act == 'relu':
            self.act_funcs.append(Network.relu_vect)
            self.act_deriv_funcs.append(Network.relu_deriv_vect)
        else:
            self.act_funcs.append(Network.sigmoid_vect)
            self.act_deriv_funcs.append(Network.sigmoid_deriv_vect)
        # add error state
        self.errors.append(np.zeros((n_neurons,), dtype=DTYPE))

    def reset_state(self, reset_inp_exp=False):
        """
        Wipe neural activation and error state (set to 0)
        @param inp_exp: additionally wipe input and expected, default is false
        """
        for lay_ind in range(len(self.layers)):
            self.layers[lay_ind].fill(0.0)
            self.errors[lay_ind].fill(0.0)
        if (reset_inp_exp):
            self.exp = None
            self.input = None

    def forward_propagate(self):
        # iterate through layers and calcualte neural activation state
        for lay_ind, lay in enumerate(self.layers):
            lay_inp = self.input # assume input is from input layer
            if (lay_ind != 0): # is it actually a deeper layer?
                lay_inp = self.layers[lay_ind-1]
            lay = np.matmul(lay_inp, self.weights[lay_ind], dtype=DTYPE) # vector multiplication for each neuron (so is matrix for layer level)
            lay = np.add(lay, self.biases[lay_ind], dtype=DTYPE) # add bias
            self.layers[lay_ind] = self.act_funcs[lay_ind](lay) # calculate activation and writeback
        # output values are at last layer

    def backward_propagate(self):
        # iterate backwards through the layers and update the error value
        # error = (expected - output) * derivative(nrn_act)
        for lay_ind in reversed(range(len(self.layers))):
            if (lay_ind == len(self.layers)-1): # is output layer
                self.errors[lay_ind] = np.subtract(self.expected, self.get_output(), dtype=DTYPE)
            else: # hidden layer
                self.errors[lay_ind] = np.matmul(self.errors[lay_ind+1], np.transpose(self.weights[lay_ind+1]), dtype=DTYPE) # errors propagate back up weights
            self.errors[lay_ind] = np.multiply(self.errors[lay_ind], self.act_deriv_funcs[lay_ind](self.layers[lay_ind]), dtype=DTYPE)

    def update_weights(self):
        for lay_ind, lay in enumerate(self.layers):
            lay_inp = None
            if lay_ind == 0:
                lay_inp = self.input 
            else:
                lay_inp = self.layers[lay_ind - 1]
            self.weights[lay_ind] += self.lrn_rate * np.outer(lay_inp, self.errors[lay_ind])
            self.biases[lay_ind] += self.lrn_rate * self.errors[lay_ind]

    def train(self, examples, truth, epochs=1, batch_size=20):
        """
        Train the network for a fixed number of epochs on a set of inputs and "true" labels
        @param inputs: (--input data dim--, n_examples) numpy array of preprocessed input data
        @param truth: (--output data dim--, n_examples) numpy array of one-hot truth labels
        Returns: summed error over the training interval
        """
        num_batches = examples.shape[-1]//batch_size # integer division, < batch_size may not be used
        for epoch in range(epochs):
            cumulative_error = 0.0
            for b_index in range(num_batches):
                for inter_batch_ind in range(batch_size): # TODO - actually implement batches
                    ex_ind = inter_batch_ind + (b_index * batch_size) # location of image in full array
                    self.set_input(examples[..., ex_ind])
                    self.set_expected(truth[..., ex_ind])
                    self.forward_propagate()
                    diff = np.subtract(self.expected, self.get_output(), dtype=DTYPE) # difference between exp and output
                    cumulative_error += np.dot(diff, diff) # sum of squared error
                    self.backward_propagate()
                    self.update_weights()
            print('Epoch: {}\nError: {}'.format(epoch+1, cumulative_error))

    def predict(self, examples):
        """
        Run inference on a set of examples and return a matrix of the output
        @param examples: numpy array of shape (--input data dim--, n_examples)
        @return predictions: numpy array of shape (--output dim--, n_examples)
        """
        predictions = np.empty(self.get_output().shape + (examples.shape[-1],), dtype=DTYPE)
        for ex_ind in range(examples.shape[-1]):
            self.set_input(examples[..., ex_ind])
            self.forward_propagate()
            predictions[..., ex_ind] = self.get_output()
        return predictions

    def evaluate(self, examples, labels):
        predictions = self.predict(examples)
        correct_count = 0
        for i in range(predictions.shape[-1]):
            ind_max = np.argmax(predictions[..., i])
            ind_max_label = np.argmax(labels[..., i])
            if (ind_max == ind_max_label):
                correct_count += 1
        return np.float64(correct_count)/np.float64(predictions.shape[-1])

    def __str__(self):
        mkstr_line = lambda np_arr, num: '{}: '.format(num) + np.array2string(np_arr, precision=4, separator=' ') + '\n'
        # print weights, activations, and biases of network
        outstr = "Network: \n"
        nrn_acts_str = "---Activations---\n"
        nrn_errors_str = "---Errors---\n"
        weights_str = "---Weights---\n"
        biases_str = "---Biases---\n"
        nrn_acts_str += mkstr_line(self.input, 'in')
        for lay_ind, lay in enumerate(self.layers):
            nrn_acts_str += mkstr_line(lay, lay_ind)
            nrn_errors_str += mkstr_line(self.errors[lay_ind], lay_ind)
            weights_str += mkstr_line(self.weights[lay_ind], lay_ind)
            biases_str += mkstr_line(self.biases[lay_ind], lay_ind)
        return outstr + nrn_acts_str + nrn_errors_str + weights_str + biases_str
