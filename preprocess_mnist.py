import numpy as np
from mnist import MNIST


mndata = MNIST('.\\mnist')
images, labels = mndata.load_training()
timages, tlabels = mndata.load_testing()

train_labels = np.zeros((10, len(labels)), dtype=np.float64)
train_images = np.empty((28*28, len(labels)), dtype=np.float64)
test_labels = np.zeros((10, len(tlabels)), dtype=np.float64)
test_images = np.empty((28*28, len(tlabels)), dtype=np.float64)
to_float = lambda x: np.float64(x/255.0)
to_float_vec = np.vectorize(to_float, otypes=[np.float64])

print("Training")
# preprocess labels, images
for i, val in enumerate(labels):
    train_labels[val][i] = 1.0
    train_images[..., i] = to_float_vec(np.array(images[i], dtype=np.float64))
    if(i%100==0):
        print(i)

print("Testing")
# preprocess test labels, images
for i, val in enumerate(tlabels):
    test_labels[val][i] = 1.0
    test_images[..., i] = to_float_vec(np.array(timages[i], dtype=np.float64))
    if(i%100==0):
        print(i)

np.save('mnist_train_images', train_images)
np.save('mnist_train_labels', train_labels)
np.save('mnist_test_images', test_images)
np.save('mnist_test_labels', test_labels)