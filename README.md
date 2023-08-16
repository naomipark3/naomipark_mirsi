TABLE OF CONTENTS:
1. DESCRIPTION OF ALL FOLDERS AND FILES IN THIS REPOSITORY
    - Border detection techniques
    - Data folders
    - Noise reduction techniques
2. DESCRIPTION OF RECOMMENDED FUTURE WORK




DESCRIPTION OF ALL FOLDERS AND FILES IN THIS REPOSITORY:

**border_detection**:
    **detection.py**: employs the a Hough Transform to detect a circular object (i.e. the planetary border) in an image. We use Astropy to read in the image, which is stored in the "FITS" format, and we use OpenCV built-in functions to convert the image to grayscale and carry out the Hough Transform on the image. Once the circular object has been found, the prediction of the object's location is drawn on top of the original image. detection.py is currently only designed to read in one image at a time. Its current implementation reads in an cleaned image called  from the "data" folder (NOTE: the current implementation of detection.py only works on images that have been destriped.). In addition, the Hough Transform (i.e. the cv2.HoughCircles method) requires for its parameters to be tuned for every different image that we use. The first three parameters of the Hough Transform should be the same for every different image that we use, but the last three will need to be tuned accordingly to the image. Here is the link to the OpenCV docs for more information about what each of the parameters in the HoughCircles method uses: https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d 

**data**:
    This folder contains a series of individual files that were used for testing and trial-and-error purposes. For example, when I initially wanted to see if my stripe noise removal algorithm would work, I applied the algorithm to one image only (instead of a folder of images). I kept that one image I was interested in cleaning in this "data" folder. 

**noise_removal**:
    **stripe_noise_indiv.py**: The stripe noise correction algorithm implemented in this code is designed to remove stripe noise artifacts from FITS images, common in astronomical data. It operates by calculating the mean value for each column of the image and then applying a second-order finite difference to identify the bias or distortion in each column. The core of the algorithm is an iterative numerical method, where a correction function applies gradient descent to update the bias term, driving the optimization process. The algorithm balances between constraint and smoothness terms and iteratively refines the estimated biases, subtracting them from the original image to achieve a denoised result. Contact naomi_park@brown.edu for a more detailed description of the theory and design behind the stripe noise removal algorithm. stripe_noise_indiv.py takes in a single image from the "data" folder and performs the algorithm on that one image. stripe_noise_indiv.py displays the original image (before stripe removal) at the start of the algorithm, and then it displays corrected image (after stripe noise removal) once the algorithm has been performed 20 times. We note that the header information of the image is NOT preserved in stripe_noise_indiv.py, as this script is primarily for visualization and visual testing purposes.  
    **stripe_noise_directory.py**: This file performs the exact same algorithm detailed in the description of stripe_noise_indiv.py above. However, instead of taking in a single image, this file takes in a directory of images. As a result, this file is more practical when we need to denoise hundreds of image after a MIRSI observing run. For example, I was able to "automatically" feed all images taken at the 4.90 micron filter on June 21st (which is the 4_90 folder within june_21) into the algorithm with the stripe_noise_directory.py file. It does so with the process_fits_file(path) method, which enables the user to specify a folder of images to destripe with the "path" variable at the end of the stripe_noise_directory.py script. The user can also specify a name and path for the corrected images to go in with the "corrected_image_path" variable. All destriped images will be dumped into the folder specified in "corrected_image_path," and each destriped image will be assigned the name: <original_image_name>_corrected.fits so that the FITS file format is preserved. Finally, unlike the stripe_noise_indiv.py file, the header data of each image is preserved (i.e. carried on to the destriped image) so that we can run the additional necessary reduction techniques on each destriped image.   





DESCRIPTION OF RECOMMENDED FUTURE WORK:

**Investigation into the source of the bias**: 
Our stripe noise removal algorithm calculates the bias in each column readout circuit of MIRSI's infrared focal plane array (contact naomi_park@brown.edu for more details on this). However, we believe that the current approach would benefit from further investigating the source of the bias. For example, what is there is variation in the bias as a function of row? Is there a pattern in where/how each individual stripe occurs? We have noticed that most images tend to have a stripe very close to the center of the image. Why is this the case? Comparing the "difference image" (i.e. the subtracted difference between the original and corrected images graphed as a 2D array with plt.imshow()) of MIRSI images taken at different micron filters and plotting the bias as a function of row and column may help to answer some of these questions.  

**Additional testing of detection.py**: The current implementation of detection.py would benefit from testing with different images. Because the Hough Transform itself requires tuning parameters, additional testing would enable us to produce a "range" of parameter values that we can try to achieve the results we want. In addition, Emma Dahl has produced code that is capable of locating the planetary limb based on the location of the telescope at the time the image was taken. Comparison of the Hough Transform with her method may help us improve the existing implementation or identify additional limitations.

**Running more trials of NEMESIS**: MIRSI was refurbished a few years ago, and the current version of NEMESIS assumes the older construction of MIRSI. We do not yet have the filter functions based on the new construction of MIRSI, but once these filter functions have been created, we may find more groundbreaking results by running additional NEMESIS trials (note: files needed for running NEMESIS are not included in this repo). 


