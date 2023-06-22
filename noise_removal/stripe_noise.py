import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from astropy.io import fits
import alphashape

im = fits.open('/Users/naomipark/Desktop/jpl_internship/naomipark_mirsi/cal_jcf.043054.gz') #reads in fits file
red_data = im[0].data #i believe this accesses pixel values of each image (not entirely sure what this means though???)
#data is in the form of I (erg/s/cm^2/ster/cm^-1)
print("original image: ", red_data[:, 200])

fig1 = plt.figure(1)
plt.imshow(im[0].data[:, 125:150])
plt.title("Original Image")
plt.show()

def mean_column(image): #will take in "im" as argument
    return np.mean(image, axis=0)

def finite_difference_second_order(b):
    return np.pad(np.diff(b,2), (1,1), 'edge')

'''
The correction function is designed to iteratively solve the Euler-Lagrange equation
via the use of a numerical method to approximate solution.
The term 'zxx - b_old_xx + lambda_ * b_old' is the LHS of my Euler-Lagrange equation.
correlation() tries to drive quantity (zxx - b_old_xx + lambda_ * b_old) towards zero by updating
the bias term in a direction that reduces this quantity (see [Eq. 6] for the implementation).

The equation that we're solving is designed to find an optimal bias 'b' that, when
subtracted from the observed image, minimizes the difference between neighboring columns
of the corrected image.

'''
def correction(b_old, z, del_t, lambda_, lambda_tv):
    z_xx = finite_difference_second_order(z)
    b_old_xx = finite_difference_second_order(b_old)
    tv_term = np.pad(np.diff(b_old), (1,0), 'constant') #**NOTE: total variation is a regularization term that encourages the resulting image
    #to have less high-frequency noise and more piecewise-constant regions
    #this will probably not help because you we want to IGNORE the piece in the middle instead of trying to account for it

    return b_old - del_t * (z_xx - b_old_xx + lambda_ * b_old) #see [Eq. 6] from article (with TV term)
    #this is the numerical approximation for the new bias 'b' (implementation of a
    #gradient descent step)

'''
'b' represents the estimated bias (i.e. offset or distortion) from the true value of each column
These biases are manifest as stripe noise. 
'''
def stripe_noise_correction(image, init_bias, del_t, niters, lambda_=0.01, lambda_tv=1):
    z = mean_column(image)
    b = init_bias

    '''in iteration, 'b' is updated to better estimate the bias of each column based on 'correction()'
    '''
    for i in range(niters):
        b = correction(b, z, del_t, lambda_, lambda_tv)
    
    ''' 'b' is subtracted from original image to correct for NU
    (i.e. subtraction makes the image more like its unbiased state)
    '''
    corrected_image = image - b #see [Eq. 7]
    return corrected_image, b


initial_bias = np.zeros(red_data[:, 125:150].shape[1]) #.shape returns a tuple that represents size, so .shape[1] helps us to access the columns
corrected_image, estimated_bias = stripe_noise_correction(red_data[:, 125:150], initial_bias, del_t=0.01, niters=1000) #set timestep equal to 0.1 and niters=100 for now
# corrected_image = denormalize(corrected_image, original_max)

fig2 = plt.figure(2)
plt.imshow(corrected_image)
plt.title("Corrected Image")
print()
# print("corrected image: ", corrected_image[:, 200])
plt.show()