import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import fftpack

# Load the image in grayscale
gray_jpg = cv2.imread('/Users/naomipark/Desktop/jpl_internship/naomipark_mirsi/corrected_img.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(gray_jpg, cmap='gray')
plt.show()

def detect():

    # Ensure the image was loaded properly
    if gray_jpg is None:
        print("Failed to load image")
    else:
        #apply Otsu's thresholding
        ret, thresh = cv2.threshold(gray_jpg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Try to find circles in the image
        plt.imshow(thresh)
        plt.show()
        #needed to tune params from: (thresh, cv2.HOUGH_GRADIENT, 1, 100, 100, 30, 30, 100)
        detected_circle = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 1000, param1 = 15, param2 = 7, minRadius=30, maxRadius=100)
        #changing num of votes doesn't do anything

    print(detected_circle)

    detected_circles = np.uint16
    if detected_circle is not None:
    
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circle))
    
        coordinate_list = []
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            coordinate_list.append((a,b))
            # Draw the circumference of the circle.
            gray_jpg_copy = gray_jpg.copy()
            cv2.circle(gray_jpg_copy, (a, b), r, (0, 255, 0), 2) #last parameter is equivalent to thickness.
    
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(gray_jpg_copy, (a, b), 1, (0, 0, 255), 3)
            cv2.imshow("Detected Circle", gray_jpg_copy)
            cv2.waitKey(0)
    return detected_circles

def generate_mask(image, circle):
    a, b, r = circle[0], circle[1], circle[2]

    y,x = np.ogrid[-b:image.shape[0]-b, -a:image.shape[1]-a]
    mask = x*x + y*y <= r*r
    
    return mask

def apply_fourier_filter(image, mask):

    F = fftpack.fftn(image) #take fft of image
    # Shift the zero-frequency component to the center of the spectrum
    F_shift = fftpack.fftshift(F)

    magnitude_spectrum = 20*np.log1p(np.abs(F_shift))

    plt.figure(figsize=(14, 7))

    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

    plt.show()

    F_shift[mask] = 0 #apply Fourier filter and zero out transform coefficients for the circle
    # Shift the zero-frequency component back to original place
    F_ishift = fftpack.ifftshift(F_shift)
    filtered_image_complex = fftpack.ifftn(F_ishift) #take inverse Fourier transform to get filtered image back in spatial domain
    filtered_image = np.abs(filtered_image_complex) # Take the absolute value to get a real image

    # Normalise the image to the range 0-255
    filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return filtered_image


detected_circles = detect()

mask = generate_mask(gray_jpg, detected_circles[0][0])
filtered_img = apply_fourier_filter(gray_jpg, mask)

cv2.imshow("Filtered Image", filtered_img)
cv2.waitKey(0)

hdu = fits.PrimaryHDU(filtered_img)
hdu.writeto("fourier_correction.fits")