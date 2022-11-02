# import all requirements   
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import scipy
from PIL import Image, ImageFilter

def main():
    
    img = cv2.imread('cubo.jpeg')
    
    # apply fourier transform
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    # show image
    cv2.imshow('image', magnitude_spectrum)
    cv2.waitKey(0)
    
    # multiply the fourier transform by a low pass gaussian filter 
    rows, cols, ch = img.shape
    crow,ccol = rows/2 , cols/2
    fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # show image
    cv2.imshow('image', img_back)
    cv2.waitKey(0)
    
    # apply high pass gausian filter
    fshift = np.fft.fftshift(f)
    fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # show image
    cv2.imshow('image', img_back)
    cv2.waitKey(0)

    
    


    




    # apply inverse fourier transform
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # show image
    cv2.imshow('image', img_back)
    cv2.waitKey(0)
    
    # close all windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    
