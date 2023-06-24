import cv2
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
plt.imshow(im[0].data)
plt.savefig('jup_output.jpg')
plt.title("Original Image")
plt.show()

gray_jpg = cv2.imread('/Users/naomipark/Desktop/jpl_internship/naomipark_mirsi/jup_output.jpg', cv2.IMREAD_GRAYSCALE)

# cv2.imshow("gray jpg", gray_jpg)
# cv2.waitKey(0)
plt.imshow(gray_jpg, cmap='gray')
plt.show()

# Ensure the image was loaded properly
if gray_jpg is None:
    print("Failed to load image")
else:
    # Try to find circles in the image
    detected_circle = cv2.HoughCircles(gray_jpg, cv2.HOUGH_GRADIENT, 145, 40, param1 = 30, param2 = 100, minRadius=50, maxRadius=150)
    #changing num of votes doesn't do anything

print(detected_circle)

'''
***GLOBAL VARS*** (not great syntax...)
'''
coordinate_list = []
radius = 0
if detected_circle is not None:
  
    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circle))

    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
        radius = r
        coordinate_list.append((a,b))
        # Draw the circumference of the circle.
        cv2.circle(gray_jpg, (a, b), r, (0, 255, 0), 2)
  
        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(gray_jpg, (a, b), 1, (0, 0, 255), 3)
        cv2.imshow("Detected Circle", gray_jpg)
        cv2.waitKey(0)

def mean_column(image, coordinate_list, radius): # Will take in "im" and "coordinate_list" as arguments
    a, b = coordinate_list[0] # Extract the x, y coordinates and radius of the planet
    r = radius
    means = []
    for j in range(image.shape[1]): # For each column
        if a-r <= j <= a+r: # If the column contains part of the planet
            upper_boundary = b - r
            lower_boundary = b + r
            above_planet = image[:upper_boundary, j] #slices image array to get all the rows above the upper boundary of the planet
            below_planet = image[lower_boundary+1:, j]
            mean = np.mean(np.concatenate([above_planet, below_planet])) #concatenate pixel values from above_planet and below_planet
            #into a single array and calculate the mean of those values
        else:
            mean = np.mean(image[:, j])
        means.append(mean)
    return np.array(means)


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
    return b_old - del_t * (z_xx - b_old_xx + lambda_ * b_old) #see [Eq. 6] from article (with TV term)
    #this is the numerical approximation for the new bias 'b' (implementation of a
    #gradient descent step)

'''
'b' represents the estimated bias (i.e. offset or distortion) from the true value of each column
These biases are manifest as stripe noise. 
'''
def stripe_noise_correction(image, init_bias, del_t, niters, lambda_=0.01):
    z = mean_column(image, coordinate_list, radius)
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
# corrected_image = denormalize(corrected_image, original_max)

fig3 = plt.figure(3)
plt.imshow(corrected_image)
plt.title("Corrected Image")
print()
plt.show()