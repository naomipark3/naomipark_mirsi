import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from astropy.io import fits

im = fits.open('/Users/naomipark/Desktop/jpl_internship/naomipark_mirsi/cal_jcf.043054.gz') #reads in fits file
red_data = im[0].data #i believe this accesses pixel values of each image (not entirely sure what this means though???)
#data is in the form of I (erg/s/cm^2/ster/cm^-1)
print("original image: ", red_data[:, 200])

fig1 = plt.figure(1)
plt.imshow(im[0].data)
plt.title("Original Image")
plt.show()

def mean_column(image): #will take in "im" as argument
    return np.mean(image, axis=0)

def finite_difference_second_order(b):
    return np.pad(np.diff(b,2), (1,1), 'edge')

# Euler-Lagrange PDE numerical approximator
def correction(b_old, z, del_t, lambda_, lambda_tv):
    z_xx = finite_difference_second_order(z)
    b_old_xx = finite_difference_second_order(b_old)
    tv_term = np.pad(np.diff(b_old), (1,0), 'constant')
    return b_old - del_t * (z_xx - b_old_xx + lambda_ * b_old + lambda_tv * tv_term) #see [Eq. 6] from article (with TV term)

def stripe_noise_correction(image, init_bias, del_t, niters, lambda_=0.00001, lambda_tv=0.1):
    z = mean_column(image)
    b = init_bias
    for i in range(niters):
        b = correction(b, z, del_t, lambda_, lambda_tv)
    corrected_image = image - b
    return corrected_image, b


def normalize(image):
    return image / np.max(image)

def denormalize(image, original_max):
    return (image * original_max).astype(original_max)

original_max = np.max(red_data)
red_data = normalize(red_data) #i can't tell if normalization/denormalization is really necessary...

initial_bias = np.zeros(red_data.shape[1]) #.shape returns a tuple that represents size, so .shape[1] helps us to access the columns
corrected_image, estimated_bias = stripe_noise_correction(red_data, initial_bias, del_t=0.01, niters=1000) #set timestep equal to 0.1 and niters=100 for now
corrected_image = denormalize(corrected_image, original_max)

fig2 = plt.figure(2)
plt.imshow(corrected_image)
plt.title("Corrected Image")
print()
print("corrected image: ", corrected_image[:, 200])
plt.show()