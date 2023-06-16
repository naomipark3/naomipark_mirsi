import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from astropy.io import fits
from gplearn.genetic import SymbolicRegressor

im = fits.open('/Users/naomipark/Desktop/jpl_internship/naomipark_mirsi/cal_jcf.043054.gz') #reads in fits file
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
y_value = np.array(y_value)

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
for i in range(len(red_data[0])):
    est = SymbolicRegressor(population_size=500, generations=10, tournament_size=20, stopping_criteria=0.01, verbose=1)
    est.fit(y_value.reshape(-1,1), red_data[:, 50])
    # equation = est.export()
    print("best fit for column: ", i, ", ", est._program)