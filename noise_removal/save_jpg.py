import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import cv2

im = fits.open('/Users/naomipark/Desktop/jpl_internship/naomipark_mirsi/data/cal_jcf02410243.gz') #reads in fits file
red_data = im[0].data
#data is in the form of I (erg/s/cm^2/ster/cm^-1)

fig1 = plt.figure(1)
plt.imshow(im[0].data)
plt.title("Original Image")
plt.show()

im2 = fits.open('/Users/naomipark/Desktop/jpl_internship/naomipark_mirsi/data/cal_jcf.043054.gz')
plt.imshow(im2[0].data)
plt.show()

# plt.imsave("corrected_img.jpg", red_data)
# grayscale_img = cv2.imread("corrected_img.jpg", cv2.IMREAD_GRAYSCALE)
# cv2.imwrite("clean_grayscale.jpg", grayscale_img) #save grayscale image
# #so it can be used in the ADMM