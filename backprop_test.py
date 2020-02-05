import numpy as np

train_labels = np.load('mnist_train_labels.npy', allow_pickle=True)
train_images = np.load('mnist_train_images.npy', allow_pickle=True)
test_labels = np.load('mnist_test_labels.npy', allow_pickle=True)
test_images = np.load('mnist_test_images.npy', allow_pickle=True)

np.random.seed(789) # randomness always the same

from network import Network

net = Network(28*28, lrn_rate=0.01)
net.add_layer(10, act='sigmoid')

print(net)

net.train(train_images, train_labels, epochs=5)
print(net)

acc = net.evaluate(test_images, test_labels) # eval on test set
print("acc: {}%".format(100*acc))