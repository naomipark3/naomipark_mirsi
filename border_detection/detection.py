import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from astropy.io import fits

# Load the image in grayscale
gray_jpg = cv2.imread('/Users/naomipark/Desktop/jpl_internship/naomipark_mirsi/data/corrected_img.jpg', cv2.IMREAD_GRAYSCALE)

plt.imshow(gray_jpg, cmap='gray')
plt.show()

# Ensure the image was loaded properly
if gray_jpg is None:
    print("Failed to load image")
else:
    #apply Otsu's thresholding
    ret, thresh = cv2.threshold(gray_jpg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Try to find circles in the image
    plt.imshow(thresh)
    plt.show()
    detected_circle = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 100, param1 = 100, param2 = 30, minRadius=30, maxRadius=100)
    #changing num of votes doesn't do anything

print(detected_circle)

if detected_circle is not None:
  
    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circle))
  
    coordinate_list = []
    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
        coordinate_list.append((a,b))
        # Draw the circumference of the circle.
        cv2.circle(gray_jpg, (a, b), r, (0, 255, 0), 2) #last parameter is equivalent to thickness.
  
        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(gray_jpg, (a, b), 1, (0, 0, 255), 3)
        cv2.imshow("Detected Circle", gray_jpg)
        cv2.waitKey(0)