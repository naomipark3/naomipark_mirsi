import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from astropy.io import fits
from gplearn.genetic import SymbolicRegressor
from scipy.optimize import curve_fit

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
# plt.plot(y_value, red_data[:, 200], label='Column 200')
# plt.xlabel('Y Value (Image Height)')
# plt.ylabel('Pixel Value')
# plt.title('Pixel Value vs. Y Value')
# plt.legend()

# Define the normal distribution function
def normal_dist(x, mu, sigma, A, D):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + D

# Perform the curve fitting
initial_guess = [np.mean(y_value), np.std(y_value), 1.0, 0.0]  # Initial parameter guess
popt, pcov = curve_fit(normal_dist, y_value, red_data[:, 200], p0=initial_guess)

# Extract the optimized parameters
mu_opt, sigma_opt, A_opt, D_opt = popt

# Generate the fitted curve
x_fit = np.linspace(1, red_data.shape[0], 100)
y_fit = normal_dist(x_fit, mu_opt, sigma_opt, A_opt, D_opt)

# Plot the original data and the fitted curve
plt.plot(y_value, red_data[:, 200], label='Column 200')
plt.plot(x_fit, y_fit, 'r-', label='Fitted Normal Distribution')
plt.xlabel('Y Value (Image Height)')
plt.ylabel('Pixel Value')
plt.title('Pixel Value vs. Y Value')
plt.legend()
plt.show()