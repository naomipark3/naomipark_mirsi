import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from astropy.io import fits

im = fits.open('/Users/naomipark/Desktop/jpl_internship/naomipark_mirsi/cal_jcf.043054.gz') #reads in fits file
red_data = im[0].data #i believe this accesses pixel values of each image (not entirely sure what this means though???)

# print("shape: ", red_data.shape) #(277, 357)
# print(red_data)
# print(red_data[:, 150]) #print columns to see where planetary border begins!
print("original image: ", red_data[:, 200])

#count the number of "cells"/pixels per each column
y_value = []
count = 0
for i in range(len(red_data[:, 200])):
    count += 1
    y_value.append(count)
# print("y_value: ", y_value)
# y_value = np.array(y_value)

fig1 = plt.figure(1)
plt.imshow(im[0].data)
plt.show()

def mean_column(image): #will take in "im" as argument
    return np.mean(image, axis=0)

def finite_difference_second_order(b):
    return np.pad(np.diff(b,2), (1,1), 'edge')

# Euler-Lagrange PDE numerical approximator
def correction(b_old, z, del_t, lambda_):
    z_xx = finite_difference_second_order(z)
    b_old_xx = finite_difference_second_order(b_old)
    return b_old - del_t * (z_xx - b_old_xx + lambda_ * b_old)

def stripe_noise_correction(image, init_bias, del_t, niters, lambda_=0.1):
    z = mean_column(image) #returns a 1-D array where each entry is the MEAN of each col
    b = init_bias
    for i in range(niters): #ok so this would be iterative and not recursive...
        b = correction(b, z, del_t, lambda_)
    corrected_image = image - b
    return corrected_image, b #should return an array of pixel values if I am correct...?

initial_bias = np.zeros(red_data.shape[1]) #.shape returns a tuple that represents size, so .shape[1] helps us to access the columns
corrected_image, estimated_bias = stripe_noise_correction(red_data, initial_bias, del_t=0.1, niters=100) #set timestep equal to 0.1 and niters=100 for now
fig2 = plt.figure(2)
plt.imshow(corrected_image)
print()
print("corrected image: ", corrected_image[:, 200])
plt.show()