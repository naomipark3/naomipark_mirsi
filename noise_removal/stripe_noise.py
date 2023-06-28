import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

im = fits.open('/Users/naomipark/Desktop/jpl_internship/naomipark_mirsi/wjup.00226.b.fits.gz') #reads in fits file
red_data = im[0].data
#data is in the form of I (erg/s/cm^2/ster/cm^-1)

fig1 = plt.figure(1)
plt.imshow(im[0].data)
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
def correction(b_old, z, del_t, lambda_):
    z_xx = finite_difference_second_order(z)
    b_old_xx = finite_difference_second_order(b_old)
    return b_old - del_t * (z_xx - b_old_xx + lambda_ * b_old) #see [Eq. 6] from article
    #this is the numerical approximation for the new bias 'b' (implementation of a
    #gradient descent step)

'''
'b' represents the estimated bias (i.e. offset or distortion) from the true value of each column
These biases are manifest as stripe noise. 
'''
def stripe_noise_correction(image, init_bias, del_t, niters, lambda_=0.1):
    z = mean_column(image)
    b = init_bias

    '''in iteration, 'b' is updated to better estimate the bias of each column based on 'correction()'
    '''
    for i in range(niters):
        b = correction(b, z, del_t, lambda_)
    
    ''' 'b' is subtracted from original image to correct for NU
    (i.e. subtraction makes the image more like its unbiased state)
    '''
    corrected_image = image - b #see [Eq. 7]
    return corrected_image, b


initial_bias = np.zeros(red_data.shape[1]) #.shape returns a tuple that represents size, so .shape[1] helps us to access the columns
corrected_image, estimated_bias = stripe_noise_correction(red_data, initial_bias, del_t=0.01, niters=1000) #set timestep equal to 0.1 and niters=100 for now

fig3 = plt.figure(3)
plt.imshow(corrected_image)
plt.title("Corrected Image")
print()
plt.show()