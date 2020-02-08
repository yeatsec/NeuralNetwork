import numpy as np

print('loading data')
train_labels = np.load('mnist_train_labels.npy', allow_pickle=True)
train_images = np.load('mnist_train_images.npy', allow_pickle=True)
test_labels = np.load('mnist_test_labels.npy', allow_pickle=True)
test_images = np.load('mnist_test_images.npy', allow_pickle=True)

train_subset_ind = 50000
test_subset_ind = 10000
filename = 'test4'

np.random.seed(123) # randomness always the same

from network import Network
print('creating network')
net = Network(28*28, lrn_rate=0.005)
net.add_layer(10, act='relu')
#net.load(filename) # later

net.train(train_images[...,:train_subset_ind], train_labels[...,:train_subset_ind], epochs=1, batch_size=1, train_acc=200, checkpoint=True)

acc = net.evaluate(test_images[...,:test_subset_ind], test_labels[...,:test_subset_ind]) # eval on test set
print("acc: {}%".format(100*acc))

net.save(filename)