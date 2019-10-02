import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt

def read_gz(filename):
    with gzip.open(filename) as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        print("Found datastructure of shape", shape)
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

def load(pwd):
    return {
        "train_images" : read_gz(pwd + "train-images-idx3-ubyte.gz"),
        "train_labels" : read_gz(pwd + "train-labels-idx1-ubyte.gz"),
        "test_images"  : read_gz(pwd + "t10k-images-idx3-ubyte.gz"),
        "test_labels"  : read_gz(pwd + "t10k-labels-idx1-ubyte.gz")}

def plot(images, labels, idxs):
    N = len(idxs)
    imagesThis = images[idxs]
    labelsThis = labels[idxs]

    fig, ax = plt.subplots(ncols=N, figsize=(2*N, 2))
    fig.suptitle("Examples from MNIST dataset")
    for i in range(N):
        ax[i].imshow(imagesThis[i])
        ax[i].set_title(str(labelsThis[i]))
    plt.show()