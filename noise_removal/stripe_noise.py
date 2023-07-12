import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import cv2
from scipy.signal import convolve2d


im = fits.open('/Users/naomipark/Desktop/jpl_internship/naomipark_mirsi/data/wjup.00226.b.fits.gz') #reads in fits file
red_data = im[0].data
#data is in the form of I (erg/s/cm^2/ster/cm^-1)

fig1 = plt.figure(1)
plt.imshow(im[0].data)
plt.title("Original Image")
plt.show()

''''
The mean_column function calculates the mean value for each column
of the image. 
@param: image -- image to denoise.
@return: numpy array where each index contains the mean value of one
column in the image.
'''
def mean_column(image, window_size=None): 
    if window_size is None:
        #if no window size specified, return mean of the entire column
        return np.mean(image, axis=0)
    else:
        #if a window size is specified, compute windowed means
        row_num, col_num = image.shape
        means = np.zeros((row_num, col_num))
        half_window = window_size // 2

        for i in range(row_num):
            #determine the start and end indices for the window
            start = max(0, i - half_window)
            end = min(row_num, i + half_window)
            #compute the mean over the window
            means[i, :] = np.mean(image[start:end, :], axis=0)

        return means

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
    if b.ndim == 1:
        # If the input is a 1D array, compute the second order finite difference as before
        diff = np.diff(b, 2)
        return np.concatenate(([0], diff, [0]))
    else:
        # If the input is a 2D array, compute the second order finite difference for each column
        row_num, col_num = b.shape
        diff = np.zeros((row_num, col_num))
        for j in range(col_num):
            diff[:, j] = np.concatenate(([0], np.diff(b[:, j], 2), [0]))
        return diff
'''
The stripe_noise_detection function detects regions with stripe noise.
It does so by computing the absolute second derivative and then 
smoothing it with a Gaussian filter. Regions where the smoothed second 
derivative exceeds a threshold are considered as containing stripe noise.
'''
def stripe_noise_detection(image, kernel_size=5, threshold=0.05):
    # Compute the second order finite difference
    diff = np.abs(finite_difference_second_order(image))

    # Create a Gaussian kernel
    kernel = cv2.getGaussianKernel(kernel_size, 0)
    kernel_2d = np.outer(kernel, kernel.transpose())

    # Convolve the difference image with the Gaussian kernel
    smoothed = convolve2d(diff, kernel_2d, mode='same', boundary='symm')

    # Create a binary mask where the smoothed difference exceeds the threshold
    mask = smoothed > threshold

    return mask

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

def local_correction(image, init_bias, del_t, niters, lambda_=0.1, window_size=None):
    # Detect regions with stripe noise
    mask = stripe_noise_detection(image)

    # Only consider pixels within the stripe noise regions for bias estimation
    masked_image = np.where(mask, image, 0)

    # Perform stripe noise correction as before, but only on the masked image
    z = mean_column(masked_image, window_size)
    b = init_bias

    for i in range(niters):
        b = correction(b, z, del_t, lambda_)

    # Apply the correction only within the stripe noise regions
    corrected_image = np.where(mask, image - b, image)

    return corrected_image, b

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
def stripe_noise_correction(image, init_bias, del_t, niters, lambda_=0.1, window_size=None):
    z = mean_column(image, window_size)
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

for i in range(20): #we run the stripe noise removal algorithm 20 times (no window size)
    corrected_image, estimated_bias = stripe_noise_correction(red_data, initial_bias, del_t=0.01, niters=1000)
    red_data = corrected_image
    initial_bias = estimated_bias

#show image after algorithm has been run x number of times
fig3 = plt.figure(3)
plt.imshow(corrected_image)
plt.title("Corrected Image (20X)")
plt.show()
plt.imsave("corrected_img.jpg", corrected_image)
grayscale_img = cv2.imread("corrected_img.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imwrite("clean_grayscale.jpg", grayscale_img) #save grayscale image
#so it can be used in the ADMM
