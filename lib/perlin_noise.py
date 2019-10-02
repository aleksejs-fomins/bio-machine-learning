import numpy as np

# Generate a bunch on random images using truncated 2D fourier basis with random coefficients
# All pixel values are non-negative
def generateImage(NX, NY, NTERMS=2):
    img = np.zeros((NX, NY))
    xx = np.array([[i for j in range(NY)] for i in range(NX)])
    yy = np.array([[j for j in range(NY)] for i in range(NX)])
    for i in range(-NTERMS, NTERMS + 1):
        for j in range(-NTERMS, NTERMS + 1):
            kx = i / NX / 2
            ky = j / NY / 2
            r0, phi0 = np.random.uniform(0, 1, 2)
            phi = kx * xx + ky * yy - phi0
            img += r0 * np.sin(2 * np.pi * phi)

    img -= np.min(img)
    img /= np.linalg.norm(img)
    return img


def generateImageCompact(nx, ny):
    # Random prefactors
    r   = np.random.uniform(0, 1,       (2 * nx + 1, 2 * ny + 1))
    phi = np.random.uniform(0, 2*np.pi, (2 * nx + 1, 2 * ny + 1))
    c = r * np.exp(1j * phi)

    # Values of wavenumber
    kxarr = (np.arange(2 * nx + 1) - nx) * (np.pi / nx)
    kyarr = (np.arange(2 * ny + 1) - ny) * (np.pi / ny)

    # kx*x, ky*y
    eikx = np.exp(1j * np.outer(kxarr, np.arange(nx)))
    eiky = np.exp(1j * np.outer(kyarr, np.arange(ny)))

    return np.real(np.einsum("ij, iu, jv", c, eikx, eiky))

def gau(x, mu, s2):
    return np.exp(-(x-mu)**2/2/s2) / (2*np.pi*s2)

def generateImgsFFT(nx, ny, detail=0.5):
    ccR   = np.random.uniform(0, 1, (nx, ny))
    ccPhi = np.random.uniform(0, 2*np.pi, (nx, ny))

    h_idx = lambda N : N//2 - np.abs(N//2 - np.arange(N))
    hIdx2D = np.outer(np.ones(nx), h_idx(ny)) + np.outer(h_idx(nx), np.ones(ny))
    hIdx2D[0] += 0.5
    hIdx2D[:, 0] += 0.5
    pref = detail**hIdx2D

    ccR *= pref
    cc = ccR * np.exp(1j*ccPhi)

    img = np.real(np.fft.ifft2(cc))
    img -= np.min(img)
    img /= np.linalg.norm(img)
    return img