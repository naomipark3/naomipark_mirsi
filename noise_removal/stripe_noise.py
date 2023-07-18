import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import cv2
import os

im = fits.open('/Users/naomipark/Desktop/jpl_internship/naomipark_mirsi/data/wjup.00059.a.fits.gz') #reads in fits file
#data is in the form of I (erg/s/cm^2/ster/cm^-1)

# fig1 = plt.figure(1)
# plt.imshow(im[0].data)
# plt.title("Original Image")
# plt.show()

''''
The mean_column function calculates the mean value for each column
of the image. 
@param: image -- image to denoise.
@return: numpy array where each index contains the mean value of one
column in the image.
'''
def mean_column(image): #will take in "im" as argument
    return np.mean(image, axis=0)

'''
The finite_difference_second_order function computes the second-order finite 
difference of the input parameter 'b.' We note that, in this implementation,
the second-order finite difference at the boundaries is set to 0 to mitigate
distortion at the edges when the algorithm is run for a large number of iterations.
@param: b -- input array that we want to calculate the second-order finite difference for.
We will either pass in an array consisting of the mean values or the estimated biases of
the columns.
@return: The second-order finite difference of the input array 'b,' where the array size
is preserved by padding a zero on either end. The second-order finite difference is returned
as a numpy array.
'''
def finite_difference_second_order(b):
    diff = np.diff(b, 2)
    return np.concatenate(([0], diff, [0]))

'''
The correction function is designed to iteratively solve the Euler-Lagrange equation
via the use of a numerical method to approximate solution.
The term 'zxx - b_old_xx + lambda_ * b_old' is the LHS of my Euler-Lagrange equation.
correlation() tries to drive quantity (zxx - b_old_xx + lambda_ * b_old) towards zero by updating
the bias term in a direction that reduces this quantity (see [Eq. 6] for the implementation).

The equation that we're solving is designed to find an optimal bias 'b' that, when
subtracted from the observed image, minimizes the difference between neighboring columns
of the corrected image.
@param: b_old -- prior estimate of the bias, which is updated per each call to correction() 
@param: z -- array containing mean intensity values for each column.
@param: del_t -- time step (regulates velocity of convergence per iteration).
@param: lambda_ -- controls the balance between the constraint term and the smoothness term.
@return: The updated bias estimate as a numpy array.
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
@param: image -- image to denoise.
@param: init_bias -- initial guess of the bias.
@param: del_t -- time step (regulates velocity of convergence per iteration).
@param: niters -- number of iterations that correction() is run.
@param: lambda_ -- controls the balance between the constraint term and the smoothness term.
@return: corrected_image -- denoised image, returned as a numpy array of intensity values.
@return: b -- updated estimate of the bias in each column, returned as a numpy array. 
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

def process_fits_file(file_path):
    im = fits.open(file_path)
    red_data = im[0].data

    initial_bias = np.zeros(red_data.shape[1]) #.shape returns a tuple that represents size, so .shape[1] helps us to access the columns

    for i in range(20): #we run the stripe noise removal algorithm 20 times
        corrected_image, estimated_bias = stripe_noise_correction(red_data, initial_bias, del_t=0.01, niters=1000)
        red_data = corrected_image
        initial_bias = estimated_bias

    #show image after algorithm has been run x number of times
    fig1 = plt.figure(1)
    plt.imshow(corrected_image)
    plt.title(f"Corrected Image (20X) - {file_path}")
    plt.show()

    #define a new path for the corrected images
    corrected_image_directory = '/Users/naomipark/Desktop/jpl_internship/naomipark_mirsi/11_70_corrected_images'
    
    #ensure the new directory exists, if not, create it
    if not os.path.exists(corrected_image_directory):
        os.makedirs(corrected_image_directory)

    # create a new filename for the corrected image
    corrected_image_filename = os.path.splitext(os.path.basename(file_path))[0] + '_corrected.jpg'
    
    # create the full path to the new file by joining the new directory with the new filename
    corrected_image_path = os.path.join(corrected_image_directory, corrected_image_filename)
    
    # save the corrected image to the new path
    plt.imsave(corrected_image_path, corrected_image)

    # # compute the difference image
    # difference_image = red_data - corrected_image

    # # plot the difference image
    # plt.figure()
    # plt.imshow(difference_image)
    # plt.title(f"Difference Image - {file_path}")
    # plt.show()


#specify your path
path = '/Users/naomipark/Desktop/jpl_internship/naomipark_mirsi/11_70/'

#lines 117-122 will be run for images in a folder for each wavelength of interest
#get list of all .fits.gz files in the directory
fits_files = [f for f in os.listdir(path) if f.endswith('.fits.gz')]

#process all files
for file in fits_files:
    full_file_path = os.path.join(path, file)
    process_fits_file(full_file_path)