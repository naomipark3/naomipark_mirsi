import numpy as np
import matplotlib.pyplot as plt
'''Class written to show different images in the dataset before 
constructing the dataset'''

img = np.load('/Users/naomipark/Desktop/jpl_internship/naomipark_mirsi/border_detection/dataset/train/0.npy')
plt.imshow(img)
plt.show()

img2 = np.load('/Users/naomipark/Desktop/jpl_internship/naomipark_mirsi/border_detection/dataset/train/519.npy')
plt.imshow(img2)
plt.show()

img3 = np.load('/Users/naomipark/Desktop/jpl_internship/naomipark_mirsi/border_detection/dataset/train/1085.npy')
plt.imshow(img3)
plt.show()

#circle border is quite faint, which I think is good?
#i think circle inside shld be filled in with noise, and 
#i think we should add stripes to images
#do we think efficacy would be increased if we used
#50 different MIRSI images and made 100 copies of each of them?
#it wld probably overfit...

#wld it help if we added some MIRSI images to training data?
#apply YOLO before trying to work with a CNN (you can train YOLO
# on your own dataset)
#is it possible to get YOLO to return the center point and
#radius of the circle?

