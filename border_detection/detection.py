import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from astropy.io import fits

#load the image in grayscale, store grayscale data in gray_jpg.
gray_jpg = cv2.imread('/Users/naomipark/Desktop/jpl_internship/naomipark_mirsi/data/corrected_img.jpg', cv2.IMREAD_GRAYSCALE)

#show image in grayscale using matplotlib. cmap='gray' specifies color scheme.
plt.imshow(gray_jpg, cmap='gray')
plt.show()

#ensure the image was loaded properly:
if gray_jpg is None:
    print("Failed to load image")

#if image has been loaded properly:
else:
    #apply Otsu's thresholding with cv2.threshold(). These parameters do NOT need to be changed,
    #even working with a different image.
    ret, thresh = cv2.threshold(gray_jpg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #show thresholded image with matplotlib.
    plt.imshow(thresh)
    plt.show()

    '''
    We find circles in the thresholded image with cv2.HoughCircles. The first three parameters of HoughCircles
    do not need to be changed, but the other five parameters may need to be tuned when working with a new image.
    Here is a description of the eight parameters in HoughCircles in respective order:
    1.image: The input single-channel 8-bit or 32-bit floating-point image, which should be a grayscale image.
    2. method: Specifies the method to detect circles. Use cv2.HOUGH_GRADIENT for the standard Hough Gradient method.
    3. dp: Inverse ratio of the accumulator resolution to the image resolution. For example, if dp = 1, the accumulator 
    has the same resolution as the input image. If dp = 2, the accumulator's resolution is half that of the input image.
    4. minDist: Minimum distance between the centers of the detected circles. This parameter helps avoid multiple circles
    being detected in close proximity.
    5. param1: The higher threshold of the two passed to the Canny edge detector. This threshold is used to find the potential edges in the image.
    6. param2: The accumulator threshold for circle detection. It is a lower threshold that filters out false circles based on the accumulator values.
    Smaller values will result in more detected circles.
    7. minRadius: Minimum radius of the circles to be detected.
    8. maxRadius: Maximum radius of the circles to be detected.
    For more an example of parameter tuning and its use, visit this site: https://pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
    '''
    detected_circle = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 100, param1 = 100, param2 = 30, minRadius=30, maxRadius=100)

#print the center coordinates and radius of the circle(s) found.
print(detected_circle)

#if a circle has been detected:
if detected_circle is not None:
  
    #convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circle))
  
    #initialize an empty list to store the circle's center coordinates.
    coordinate_list = []
    for pt in detected_circles[0, :]: #iterate through each detected circle's parameters.
        a, b, r = pt[0], pt[1], pt[2] #extract circle's center coordinates (a,b) and radius (r).
        coordinate_list.append((a,b)) #add center coordinates to the coordinate_list.
        #draw the circumference of the circle.
        cv2.circle(gray_jpg, (a, b), r, (0, 255, 0), 2) #last parameter is equivalent to thickness.
  
        #draw a small circle (of radius 1) to show the center.
        cv2.circle(gray_jpg, (a, b), 1, (0, 0, 255), 3)

        #show the detected circle on top of the grayscale image.
        cv2.imshow("Detected Circle", gray_jpg)

        #the image will stay open until the '0' key is pressed.
        cv2.waitKey(0)