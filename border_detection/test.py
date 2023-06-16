from numpy import asarray
from PIL import Image
import matplotlib.pyplot as plt
from astropy.io import fits

image = Image.open("/home/npark/Desktop/naomipark_mirsi/feb_28_sample.png")
data = asarray(image)
red_data = data[:,:,0] #transforms 'data' into a 2D array;
#access red value only for each pixel

im = fits.open('/home/npark/Desktop/naomipark_mirsi/jds.229.a.fits.gz') #this works too!

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
plt.imshow(image) #display original feb 28 image in in dimensions of (282, 365)

# for i in range(red_data.shape[0]):
#     for j in range(red_data.shape[1]):
#         r_value = red_data[i,j]
#         plt.text(j,i,str(r_value), color='black', fontsize=6, ha='center', va='center')

plt.show()

fig2 = plt.figure(2)
plt.plot(y_value, red_data[:, 200], label='Column 200')
plt.xlabel('Y Value')
plt.ylabel('Red Pixel Value')
plt.title('Red Value vs. Y Value')
plt.legend()

plt.show()
