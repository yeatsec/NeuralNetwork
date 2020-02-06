# Eric Yeats
# February 2, 2020
# Complex Neural Network Implementation
# Simple Multi-Layer Perceptron Interface with Complex Backend + Backpropagation Algorithm
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
        self.n_inputs = n_inputs
        self.input = np.zeros((n_inputs, 2), dtype=DTYPE)
        self.expected = None
        self.layers = list()
        self.weights = list()
        self.biases = list()
        # self.act_funcs = list()
        # self.act_deriv_funcs = list()
        self.errors = list()
        self.errors_batch = list()
        self.readout = None
        self.readout_wgt = None
        self.lrn_rate = lrn_rate

    def set_input(self, input_data):
        if (type(input_data) != type(self.input)):
            raise ValueError('input data must be numpy ndarray')
        for i in range(self.n_inputs):
            self.input[i] = np.array([input_data[i], 0.0] ,dtype=DTYPE)

    def set_expected(self, expected_out_data):
        self.expected = np.zeros((len(expected_out_data), 2), dtype=DTYPE)
        for i in range(len(expected_out_data)):
            self.expected[i][0] = expected_out_data[i]

    def get_output(self):
        return self.readout

    def get_output_scalar(self):
        return Network.comp_mag_vect(Network.relu_comp_vect(self.get_output()))

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
            act_deriv = 0.00001
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

    @staticmethod
    def relu_comp(nrn_comp_preact):
        if (nrn_comp_preact.shape != (2,)):
            raise ValueError('inbound shape of relu comp must be 2-vec (column) but was {}'.format(nrn_comp_preact.shape))
        mag = ((nrn_comp_preact[0]**2) + (nrn_comp_preact[1]**2))**0.5
        if (mag <= 1.0 and mag != 0.0):
            return nrn_comp_preact
        else:
            return nrn_comp_preact/mag

    @staticmethod
    def relu_comp_vect(in_arr):
        out_arr = np.empty_like(in_arr)
        # iterate through the array[..., i] and apply relu_comp to each of the elements
        for nrn_ind, nrn_preact in enumerate(in_arr):
            out_arr[nrn_ind] = Network.relu_comp(nrn_preact)
        return out_arr

    @staticmethod
    def relu_comp_deriv(nrn_comp_preact):
        if (nrn_comp_preact.shape != (2,)):
            raise ValueError('inbound shape of relu comp must be 2-vec (column) but was {}'.format(nrn_comp_preact.shape))
        mag = ((nrn_comp_preact[0]**2) + (nrn_comp_preact[1]**2))**0.5
        if (mag <= 1.0 and mag != 0.0):
            return np.ones((2,), dtype=DTYPE)
        else:
            return np.full((2,), 0.001, dtype=DTYPE)

    @staticmethod
    def relu_comp_deriv_vect(in_arr):
        out_arr = in_arr.copy()
        # iterate through the array[..., i] and apply relu_comp to each of the elements
        for nrn_ind, nrn_preact in enumerate(in_arr):
            out_arr[nrn_ind] = Network.relu_comp_deriv(nrn_preact)
        return out_arr

    def add_layer(self, n_neurons, act='sigmoid'):
        # add to parallel weight list
        if (len(self.layers) != 0): # not first layer added
            stup = (self.layers[-1].shape[0], n_neurons, 2)
            self.weights.append(np.subtract(np.random.rand(stup[0], stup[1], stup[2])*0.1, \
                np.full(stup, 0.05, dtype=DTYPE)))
        else: # the first layer added
            stup = (self.n_inputs, n_neurons, 2)
            self.weights.append(np.subtract(np.random.rand(stup[0], stup[1], stup[2])*0.1, \
                np.full(stup, 0.05, dtype=DTYPE)))
        # add to parallel bias list
        self.biases.append(np.subtract(np.random.rand(n_neurons, 2)*0.2, np.full((n_neurons, 2), 0.1, dtype=DTYPE)))
        # add layer state
        self.layers.append(np.zeros((n_neurons, 2), dtype=DTYPE))
        # keep track of the activation function for the layer
        # self.act_funcs.append(Network.relu_comp_vect)
        # self.act_deriv_funcs.append(Network.relu_comp_deriv_vect)
        # add error state
        self.errors.append(np.zeros((n_neurons, 2), dtype=DTYPE))
        self.errors_batch.append(np.zeros((n_neurons, 2), dtype=DTYPE))

    def set_readout(self):
        if (len(self.layers) == 0):
            raise ValueError('must have at least one layer added to add the readout layer')
        stup = self.layers[-1].shape
        self.readout_wgt = np.array([[1.0, 0.0] for i in range(stup[0])], dtype=DTYPE) # initialize with unit mag, no rotation
        self.readout = np.zeros_like(self.layers[-1], dtype=DTYPE)
        self.errors.append(np.zeros(stup, dtype=DTYPE))
        self.errors_batch.append(np.zeros(stup, dtype=DTYPE))

    def reset_state(self, reset_inp_exp=False):
        """
        Wipe neural activation and error state (set to 0)
        @param inp_exp: additionally wipe input and expected, default is false
        """
        for lay_ind in range(len(self.layers)):
            self.layers[lay_ind].fill(0.0)
            self.errors[lay_ind].fill(0.0)
            self.errors_batch[lay_ind].fill(0.0)
        if (reset_inp_exp):
            self.expected = None
            self.input = None

    @staticmethod
    def make_rot_mat(comp_num, complex_conj=False):
        if complex_conj:
            return np.array([[comp_num[0], comp_num[1]], [-1.*comp_num[1], comp_num[0]]], dtype=DTYPE)
        else:
            return np.array([[comp_num[0], -1.*comp_num[1]], [comp_num[1], comp_num[0]]], dtype=DTYPE) # adjusted 2nd row

    @staticmethod
    def matmul_comp(vec, mat):
        """
        vec: (n, 2)
        mat: (n, m, 2)
        vec x mat --> (m, 2)
        """
        out_arr = np.zeros((mat.shape[1], 2), dtype=DTYPE)
        for n_i, vec_val in enumerate(vec):
            for m_i, mat_val in enumerate(mat[n_i]):
                out_arr[m_i] += Network.mult_comp(vec_val, mat_val)
        return out_arr

    @staticmethod
    def mult_comp(c1, c2, complex_conj=False):
        return np.matmul(Network.make_rot_mat(c2, complex_conj=complex_conj), c1)

    @staticmethod
    def mult_comp_vect(v1, v2, complex_conj=False):
        outarr = np.empty_like(v1, dtype=DTYPE)
        for i, val in enumerate(v1):
            outarr[i] = Network.mult_comp(val, v2[i], complex_conj=complex_conj)
        return outarr

    def forward_propagate(self):
        # iterate through layers and calcualte neural activation state
        for lay_ind, lay in enumerate(self.layers):
            lay_inp = self.input # assume input is from input layer
            if (lay_ind != 0): # is it actually a deeper layer?
                lay_inp = self.layers[lay_ind-1]
            lay = Network.matmul_comp(lay_inp, self.weights[lay_ind]) # vector multiplication for each neuron (so is matrix for layer level)
            lay = np.add(lay, self.biases[lay_ind], dtype=DTYPE) # add bias
            self.layers[lay_ind] = Network.relu_comp_vect(lay)#self.act_funcs[lay_ind](lay) # calculate activation and writeback
        # output values must be propagated through readout
        self.readout = Network.relu_comp_vect(Network.mult_comp_vect(self.layers[-1], self.readout_wgt))

    @staticmethod
    def transpose_comp_2d(in_array, complex_conj=True):
        """
        transpose all complex numbers in the matrix
        if complex_conj, also negate the imaginary part of each number
        """
        shape = in_array.shape[:-1][::-1] + (2,) # reverse non-complex dimensions and add 2 back at end
        out_arr = np.empty(shape, dtype=DTYPE)
        for row in range(in_array.shape[0]):
            for col in range(in_array.shape[1]):
                out_arr[col][row] = in_array[row][col]
                if (complex_conj):
                    out_arr[col][row][1] *= -1.0 # negate imaginary part (there are probably cooler ways to do this)
        return out_arr

    def backward_propagate(self):
        # readout error
        self.errors[-1] = np.subtract(self.expected, self.get_output(), dtype=DTYPE)
        # iterate backwards through the layers and update the error value
        # error = (expected - output) * derivative(nrn_act)
        for lay_ind in reversed(range(len(self.layers))):
            if (lay_ind == len(self.layers)-1): # is last hidden layer
                self.errors[lay_ind] = Network.mult_comp_vect(self.errors[-1], self.readout_wgt, complex_conj=True)    # propagate error back from readout
            else: # hidden layer
                self.errors[lay_ind] = Network.matmul_comp(self.errors[lay_ind+1], Network.transpose_comp_2d(self.weights[lay_ind+1])) # errors propagate back up weights
            self.errors[lay_ind] = np.multiply(self.errors[lay_ind], Network.relu_comp_deriv_vect(self.layers[lay_ind]), dtype=DTYPE) # self.act_deriv_funcs

    @staticmethod
    def outer_comp(vec1, vec2):
        out_arr = np.empty((vec1.shape[0], vec2.shape[0], 2), dtype=DTYPE)
        for i1, c1 in enumerate(vec1):
            for i2, c2 in enumerate(vec2):
                out_arr[i1][i2] = Network.mult_comp(c1, c2)
        return out_arr

    def update_weights(self):
        # update readout weights
        self.readout_wgt += self.lrn_rate * Network.mult_comp_vect(self.layers[-1], self.errors_batch[-1])
        # enforce unit magnitude
        self.readout_wgt = Network.make_unit_vect(self.readout_wgt)
        # update the other weights
        for lay_ind, lay in enumerate(self.layers):
            lay_inp = None
            if lay_ind == 0:
                lay_inp = self.input 
            else:
                lay_inp = self.layers[lay_ind - 1]
            self.weights[lay_ind] += self.lrn_rate * Network.outer_comp(lay_inp, self.errors_batch[lay_ind])
            self.biases[lay_ind] += self.lrn_rate * self.errors_batch[lay_ind]

    @staticmethod
    def comp_mag(comp_num):
        return np.dot(comp_num, comp_num)**0.5

    @staticmethod
    def comp_ang(comp_num):
        return np.arctan(comp_num[1]/comp_num[0], dtype=DTYPE)

    @staticmethod
    def make_unit_vect(comp_num_arr):
        for i, comp_num in enumerate(comp_num_arr):
            comp_num_arr[i] = comp_num/Network.comp_mag(comp_num)
        return comp_num_arr
        

    @staticmethod
    def comp_mag_vect(arr):
        outarr = np.empty((arr.shape[0],), dtype=DTYPE)
        for i, comp_num in enumerate(arr):
            outarr[i] = Network.comp_mag(comp_num)
        return outarr

    @staticmethod
    def comp_ang_vect(arr):
        outarr = np.empty((arr.shape[0],), dtype=DTYPE)
        for i, comp_num in enumerate(arr):
            outarr[i] = Network.comp_ang(comp_num)
        return outarr

    def transfer_batch_error(self, batch_size):
        for i in range(len(self.errors_batch)):
            self.errors_batch[i] += self.errors[i]/batch_size

    def reset_batch_error(self):
        for i in range(len(self.errors_batch)):
            self.errors_batch[i].fill(0.0)

    def train(self, examples, truth, epochs=1, batch_size=20, verbose=True, train_acc=0, report=False):
        """
        Train the network for a fixed number of epochs on a set of inputs and "true" labels
        @param inputs: (--input data dim--, n_examples) numpy array of preprocessed input data
        @param truth: (--output data dim--, n_examples) numpy array of one-hot truth labels
        Returns: summed error over the training interval
        """
        num_batches = examples.shape[-1]//batch_size # integer division, some num < batch_size may not be used
        for epoch in range(epochs):
            cumulative_error = 0.0
            for b_index in range(num_batches):
                self.reset_batch_error()
                if verbose and b_index % 100 == 0:
                    print('Epoch {}: Batch {} of {}'.format(epoch+1, b_index+1, num_batches))
                for inter_batch_ind in range(batch_size):
                    ex_ind = inter_batch_ind + (b_index * batch_size) # location of image in full array
                    self.set_input(examples[..., ex_ind])
                    self.set_expected(truth[..., ex_ind])
                    self.forward_propagate()
                    diff = np.subtract(self.expected, self.get_output(), dtype=DTYPE) # difference between exp and output
                    diff = Network.comp_mag_vect(diff)
                    cumulative_error += np.dot(diff, diff) # sum of squared error
                    self.backward_propagate()
                    self.transfer_batch_error(batch_size)
                self.update_weights()
            print('Epoch: {}\nError: {}'.format(epoch+1, cumulative_error))
            if (train_acc > 0):
                print('Training Accuracy: {}%'.format(self.evaluate(examples[...,:train_acc], truth[...,:train_acc])*100.0))
            if (report):
                self.report()

    def predict(self, examples):
        """
        Run inference on a set of examples and return a matrix of the output
        @param examples: numpy array of shape (--input data dim--, n_examples)
        @return predictions: numpy array of shape (--output dim--, n_examples)
        """
        predictions = np.empty(self.get_output_scalar().shape + (examples.shape[-1],), dtype=DTYPE)
        for ex_ind in range(examples.shape[-1]):
            self.set_input(examples[..., ex_ind])
            self.forward_propagate()
            predictions[..., ex_ind] = self.get_output_scalar()
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
        readout_wgt_str = "---Readout Weights---\n"
        nrn_acts_str += mkstr_line(self.input, 'in')
        readout_wgt_str += mkstr_line(self.readout_wgt, 'ROW')
        for lay_ind, lay in enumerate(self.layers):
            nrn_acts_str += mkstr_line(lay, lay_ind)
            nrn_errors_str += mkstr_line(self.errors[lay_ind], lay_ind)
            weights_str += mkstr_line(self.weights[lay_ind], lay_ind)
            biases_str += mkstr_line(self.biases[lay_ind], lay_ind)
        nrn_acts_str += mkstr_line(self.get_output_scalar(), 'RO')
        return outstr + nrn_acts_str + nrn_errors_str + weights_str + readout_wgt_str + biases_str

    def report(self, classnum=1):
        import matplotlib.pyplot as plt
        for lay_ind, lay in enumerate(self.layers):
            plt.figure()
            plt.title('Layer {} Class {} Weights Magnitude'.format(lay_ind, classnum))
            plt.imshow(Network.comp_mag_vect(self.weights[lay_ind][..., classnum, :]).reshape((28, 28)), vmin=0.0, vmax=1.0)
            plt.figure()
            plt.title('Layer {} Class {} Weights Angle'.format(lay_ind, classnum))
            plt.imshow(Network.comp_ang_vect(self.weights[lay_ind][..., classnum, :]).reshape((28, 28)), vmin=-np.pi, vmax=np.pi)
        plt.show()

    def save(self, filename, path='.\\'):
        import json
        from dill.source import getsource
        print("Saving Network to {}".format(path+filename))
        savedict = {}
        savedict['filename'] = filename
        savedict['n_inputs'] = self.n_inputs
        savedict['input'] = self.input.tolist()
        if (self.expected is not None):
            savedict['expected'] = self.expected.tolist()
        savedict['layers'] = [lay.tolist() for lay in self.layers]
        savedict['weights'] = [wgt.tolist() for wgt in self.weights]
        savedict['biases'] = [b.tolist() for b in self.biases]
        # savedict['act_funcs'] = [getsource(f) for f in self.act_funcs]
        # savedict['act_deriv_funcs'] = [getsource(f) for f in self.act_deriv_funcs]
        savedict['errors'] = [err.tolist() for err in self.errors]
        savedict['errors_batch'] = [err.tolist() for err in self.errors_batch]
        if (self.readout is not None):
            savedict['readout'] = self.readout.tolist()
            savedict['readout_wgt'] = self.readout_wgt.tolist()
        savedict['lrn_rate'] = self.lrn_rate
        with open('{}.txt'.format(path+filename), 'w') as outfile:
            json.dump(savedict, outfile)
        print("Save Finished!")

    def load(self, filename, path='.\\'):
        import json
        print("Loading Network from {}".format(path+filename))
        with open(path+filename+'.txt') as json_file:
            savedict = json.load(json_file)
            self.n_input = savedict['n_inputs']
            self.input = np.array(savedict['input'], dtype=DTYPE)
            self.expected = None
            if 'expected' in savedict:
                self.expected = np.array(savedict['expected'], dtype=DTYPE)
            self.layers = [np.array(lay, dtype=DTYPE) for lay in savedict['layers']]
            self.weights = [np.array(wgt, dtype=DTYPE) for wgt in savedict['weights']]
            self.biases = [np.array(b, dtype=DTYPE) for b in savedict['biases']]
            # self.act_funcs = savedict['act_funcs']
            # self.act_deriv_funcs = savedict['act_deriv_funcs']
            self.errors = [np.array(err, dtype=DTYPE) for err in savedict['errors']]
            self.errors_batch = [np.array(err, dtype=DTYPE) for err in savedict['errors_batch']]
            self.readout = np.array(savedict['readout'], dtype=DTYPE)
            self.readout_wgt = np.array(savedict['readout_wgt'], dtype=DTYPE)
            self.lrn_rate = savedict['lrn_rate']
        print("Load Finished!")