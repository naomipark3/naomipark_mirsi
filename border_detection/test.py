from numpy import asarray
from PIL import Image
import matplotlib.pyplot as plt
from astropy.io import fits

# image = Image.open("/home/npark/Desktop/naomipark_mirsi/feb_28_sample.png")
# data = asarray(image)
# red_data = data[:,:,0] #transforms 'data' into a 2D array;
# #access red value only for each pixel

im = fits.open('/home/npark/Desktop/naomipark_mirsi/cal_jcf.043054.gz') #this works too!
red_data = im[0].data #i believe this accesses pixel values of each image

# print(red_data.shape)
# print(red_data)
print(red_data[:, 150]) #print columns to see where planetary border begins!
print(red_data[:, 200])

#count the number of "cells"/pixels per each column
y_value = []
count = 0
for i in range(len(red_data[:, 150])):
    count += 1
    y_value.append(count)
#we have 282 pix/col (makes sense
#because there are 282 rows)
print(y_value)

fig1 = plt.figure(1)
plt.imshow(im[0].data) #display original feb 28 image in in dimensions of (282, 365)

plt.show()

fig2 = plt.figure(2)
plt.plot(y_value, red_data[:, 200], label='Column 200')
plt.xlabel('Y Value')
plt.ylabel('Red Pixel Value')
plt.title('Red Value vs. Y Value')
plt.legend()

plt.show()
