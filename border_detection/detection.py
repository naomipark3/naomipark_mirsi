import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from astropy.io import fits

im = fits.open('/Users/naomipark/Desktop/jpl_internship/naomipark_mirsi/cal_jcf.043054.gz') #reads in fits file
red_data = im[0].data #i believe this accesses pixel values of each image (not entirely sure what this means though???)
#data is in the form of I (erg/s/cm^2/ster/cm^-1)
fig1 = plt.figure(1)
plt.imshow(im[0].data)
plt.savefig('jup_output.jpg')
plt.title("Original Image")
plt.show()

# Load the image in grayscale
gray_jpg = cv2.imread('/Users/naomipark/Desktop/jpl_internship/naomipark_mirsi/noise_removal/jup_output.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow("gray jpg", gray_jpg)
cv2.waitKey(0)
# Ensure the image was loaded properly
if gray_jpg is None:
    print("Failed to load image")
else:
    # Try to find circles in the image
    detected_circle = cv2.HoughCircles(gray_jpg, cv2.HOUGH_GRADIENT, 150, 40, param1 = 30, param2 = 100, minRadius=30, maxRadius=200)
    #150 yields best results so far
    #changing num of votes doesn't do anything

print(detected_circle)

if detected_circle is not None:
  
    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circle))
  
    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
  
        # Draw the circumference of the circle.
        cv2.circle(gray_jpg, (a, b), r, (0, 255, 0), 2)
  
        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(gray_jpg, (a, b), 1, (0, 0, 255), 3)
        cv2.imshow("Detected Circle", gray_jpg)
        cv2.waitKey(0)