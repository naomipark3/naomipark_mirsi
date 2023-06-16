from numpy import asarray
from PIL import Image
import matplotlib.pyplot as plt
from astropy.io import fits

im = fits.open('/home/npark/Desktop/naomipark_mirsi/cal_jcf.043054.gz') #reads in fits file
red_data = im[0].data #i believe this accesses pixel values of each image (not entirely sure what this means though???)

print("shape: ", red_data.shape) #(277, 357)
# print(red_data)
# print(red_data[:, 150]) #print columns to see where planetary border begins!
print(red_data[:, 200])

#count the number of "cells"/pixels per each column
y_value = []
count = 0
for i in range(len(red_data[:, 200])):
    count += 1
    y_value.append(count)
print("y_value: ", y_value)

fig1 = plt.figure(1)
plt.imshow(im[0].data)

plt.show()

fig2 = plt.figure(2)
plt.plot(y_value, red_data[:, 200], label='Column 200')
plt.xlabel('Y Value (Image Height)')
plt.ylabel('Pixel Value')
plt.title('Pixel Value vs. Y Value')
plt.legend()

plt.show()
