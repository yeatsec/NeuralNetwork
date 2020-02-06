import numpy as np

print('loading data')
train_labels = np.load('mnist_train_labels.npy', allow_pickle=True)
train_images = np.load('mnist_train_images.npy', allow_pickle=True)
test_labels = np.load('mnist_test_labels.npy', allow_pickle=True)
test_images = np.load('mnist_test_images.npy', allow_pickle=True)

train_subset_ind = 1000
test_subset_ind = 200

np.random.seed(789) # randomness always the same

from network import Network
print('creating network')
net = Network(28*28, lrn_rate=0.05)
net.add_layer(10, act='relu')

print(net)

net.train(train_images[...,:train_subset_ind], train_labels[...,:train_subset_ind], epochs=5)
print(net)

acc = net.evaluate(test_images[...,:test_subset_ind], test_labels[...,:test_subset_ind]) # eval on test set
print("acc: {}%".format(100*acc))