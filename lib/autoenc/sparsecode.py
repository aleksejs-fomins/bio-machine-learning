'''
Based on Olshausen & Field - https://www.nature.com/articles/381607a0.pdf

  I' ~ sum_i (a_i * f_i)
  * f_i - basis functions
  * a_i - Coefficients

  L = |I - I'|^2 + (l\s2) * sum_i |a_i|_1

  tau * df_k\dt = -1/2 dL/df_k = | I-I'><a_k >

  tau * da_k\dt = -1/2 dL/da_k = |f_k >< I-I'| + (l\s2) sgn(a_k)

Comments:
    * So this is effectively same as RB - matrix is optimized to improve shallow reconstruction error
    * Like PCA subspace, but not orthogonal, because orthogonality does not achieve anything additional here
    *

Problems:
    * Does not work without whitening of data - not sure why - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7808140
    * What's the point of sparsity on coefficients?? I would have expected sparsity of the basis functions...

TODO:
    * Try whitening
    * Try adding white noise during training, hopefully it optimizes on average image
    * Try Oja and Sanger's rules - https://en.wikipedia.org/wiki/Generalized_Hebbian_Algorithm
'''

#############################
# Imports
#############################
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
import pathlib

#############################
# Get some pictures from standard TF databases
#############################
data_dir = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', fname='flower_photos', untar=True)
data_dir = pathlib.Path(data_dir)
imgPaths = list(data_dir.glob('*/*.jpg'))
print("Total images", len(imgPaths))

# Convert all images to the same size
# The 1./255 is to convert from uint8 to float32 in range [0,1].
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

BATCH_SIZE = 100
IMG_HEIGHT = 224
IMG_WIDTH = 224
PIX_TOT = IMG_HEIGHT * IMG_WIDTH

train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH))

# Display images
# def show_batch(image_batch):
#   plt.figure(figsize=(10,10))
#   for n in range(25):
#       ax = plt.subplot(5,5,n+1)
#       plt.imshow(image_batch[n])
#       plt.axis('off')
#
# image_batch, _ = next(train_data_gen)
# show_batch(image_batch)
# plt.show()

# Get an image batch, make grayscale, flatten
image_batch, _ = next(train_data_gen)
image_batch = np.mean(image_batch, axis=-1)
image_batch_1D = image_batch.reshape((BATCH_SIZE, PIX_TOT))


#############################
# Procedures
#############################

def sigmoid(x):
    return 2 / (1 + np.exp(-x)) - 1

def relVecErr(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(a)

# bf [nPix, nBF]
def converge_coeff(img, bf, coeff, ldivs2, etaCoeff):
    M1 = img.dot(bf)
    M2 = bf.T.dot(bf)

    c = np.copy(coeff)
    i = 0
    while True:
        i += 1
        cnew = c + etaCoeff * (M1 - M2.dot(c) - ldivs2 * sigmoid(c)) #np.sign(c))
        conv = relVecErr(c, cnew)
        c = cnew

        if conv < 1.0E-3:
            return cnew, i
        if i > 1000:
            print("Warning: After", i, "iterations failed to converge. Rel error =", conv)
            return cnew, i



def step_bf(img, bf, coeff, etaBF):
    return bf + etaBF * np.outer(img - bf.dot(coeff), coeff)


##############################
# Define parameters
##############################

etaBF = 0.1
etaCoeff = 0.00001
nBF = 121  # number of basis functions
ldivs2 = 0.14 / np.std(image_batch_1D)

coeff = np.random.normal(0,1, nBF)
bf = np.random.normal(0,1, (PIX_TOT, nBF))


##############################
# Run simulation
##############################

bfRelErrLst = []


#print("Current Epoch", iEpoch)
for iImg, img in enumerate(image_batch_1D):
    for iEpoch in range(100):
        coeff, iterCount = converge_coeff(img, bf, coeff, ldivs2, etaCoeff)
        bfnew = step_bf(img, bf, coeff, etaBF)
        bfRelErrLst += [relVecErr(bf, bfnew)]
        bf = bfnew
        print("Did image", iImg, "using", iterCount, "iterations. RelBFErr =", bfRelErrLst[-1])

plt.figure()
plt.plot(bfRelErrLst)
plt.title("Relative BF change for each image")
plt.show()
##############################
# Plot resulting basis functions
##############################
plt.figure(figsize=(15,15))
for iBF in range(nBF):
    ax = plt.subplot(11, 11, iBF+1)
    plt.imshow(bf[:, iBF].reshape(IMG_WIDTH, IMG_HEIGHT))
    plt.axis('off')
plt.show()
