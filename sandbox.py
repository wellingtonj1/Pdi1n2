# libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussFilterLowPass(shape, D0):
    M,N = shape
    H = np.zeros((M,N), dtype=np.float32)
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            H[u,v] = np.exp(-D**2/(2*D0*D0))
            
    return H

def gaussFilterHighPass(shape, D0):
    M,N = shape
    H = np.zeros((M,N), dtype=np.float32)
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            H[u,v] = 1 - np.exp(-D**2/(2*D0*D0))
            
    return H

def printInCmapGray5x5(img, title):
    plt.figure(figsize=(5,5))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(title)
    plt.show()
    
    # save image
    cv2.imwrite(title + '.jpeg', img)
    

# open the image f
f = cv2.imread('cubo.jpeg',0)
printInCmapGray5x5(f, 'f(x, y)')


# transform the image into frequency domain, f --> F
F = np.fft.fft2(f)
Fshift = np.fft.fftshift(F)

printInCmapGray5x5(np.log1p(np.abs(F)), 'F(u, v)')
fshifted = np.log1p(np.abs(Fshift))
printInCmapGray5x5(fshifted, 'F(u, v) inversa')


# Gaussian: low pass filter
D0 = 10
Huv = gaussFilterLowPass(f.shape, D0)
printInCmapGray5x5(Huv, 'D0 = 10, low pass filter')

# Gaussian: High pass filter
HPF10 = 1 - Huv
printInCmapGray5x5(HPF10, 'D0 = 10, high pass filter')

# low pass d0=50
D0 = 50
H = gaussFilterLowPass(f.shape, D0)
printInCmapGray5x5(H, 'D0 = 50, low pass filter')

# Gaussian: High pass filterd d0=50
HPF = 1 - H
printInCmapGray5x5(HPF, 'D0 = 50, high pass filter')

D0 = 10
H = gaussFilterLowPass(f.shape, D0)

# Image Filters
Gshift = fshifted * Huv
G = np.fft.ifftshift(Gshift)
g = np.abs(np.fft.ifft2(G))

printInCmapGray5x5(np.log1p(np.abs(Gshift)), 'G(u, v) inversa') 
printInCmapGray5x5(np.log1p(np.abs(G)), 'G(u, v)')

# apply g * inverse g
gxy = np.log1p(np.abs(Gshift)) * np.log1p(np.abs(G))
printInCmapGray5x5(gxy, 'g(x, y)')

# apply HPF10 * fshifted
HPF10fshifted = HPF10 * fshifted
printInCmapGray5x5(HPF10fshifted, 'HPF10 * fshifted')

# Image Filters
Gshift = Fshift * HPF
G = np.fft.ifftshift(Gshift)
g = np.abs(np.fft.ifft2(G))

printInCmapGray5x5(g, 'G(u, v)')
