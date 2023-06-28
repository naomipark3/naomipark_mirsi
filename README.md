This project is organized as follows:

**border_detection**:
detection.py: employs the a Hough Transform to detect a circular object (i.e. the planetary border) in an image. We use Astropy to read in the image, which is stored in the "FITS" format, we use an OpenCV to convert the image to grayscale and carry out the Hough Transform on the image. Once the circular object has been found, the prediction of the object's location is drawn on top of the original image.

**noise_removal**:
stripe_noise.py: iteratively reduces stripe noise through an approach proposed by Shu-Peng Wang. The image, which should be stored in the FITS format, is read in by the Astropy package. We break it down into a 2D array of intensity (erg/s/cm^2/ster/cm^-1) values, and the image is fed into the stripe_noise_correction() method, where the stripe noise is removed. Additional details regarding the algorithm and its implementation are provided in the research report and the class/method header comments. 
