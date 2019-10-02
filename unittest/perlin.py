import numpy as np
import matplotlib.pyplot as plt
from lib.perlin_noise import generateImage, generateImageCompact, generateImgsFFT

nImgs = 20
nPix = 20
imgs = [generateImgsFFT(20, 20, detail=0.5) for i in range(nImgs)]

fig, ax = plt.subplots(ncols=nImgs)
for i in range(nImgs):
    ax[i].imshow(imgs[i])

plt.show()


# fig, ax = plt.subplots(nrows=nImgs, ncols=nImgs)
#
# imgs = np.zeros((nImgs, nImgs, nPix, nPix))
# for i in range(nImgs):
#     for j in range(nImgs):
#         #idxPhi = np.random.uniform(0, 2 * np.pi, nPix)
#         idxR = np.zeros((nPix, nPix))
#         idxR[i, j] = 1
#         imgs[i, j] = np.imag(np.fft.ifft2(idxR))# + np.exp(1j * idxPhi))
#
#         ax[i, j].imshow(imgs[i, j])
#
# plt.show()