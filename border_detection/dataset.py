import numpy as np
#from skimage.draw import circle_perimeter_aa
import csv
import random
from skimage.draw import disk

def draw_circle(img, row, col, rad):
    rr, cc = disk((row, col), rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = 1

def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    # Only add noise to the circle itself
    img += random.uniform(0.01, 1) * noise * np.random.rand(*img.shape)
    return (row, col, rad), img




def train_set():
    number_of_images = 200000
    level_of_noise = 3.5
    with open("train_set.csv", 'w', newline='') as outFile:
        header = ['NAME', 'ROW', 'COL', 'RAD']
        write(outFile, header)
        for i in range(number_of_images):
            params, img = noisy_circle(200, 100, level_of_noise)
            np.save("/Users/naomipark/Desktop/jpl_internship/naomipark_mirsi/border_detection/dataset/train/" + str(i) + ".npy", img)
            write(outFile, ["/Users/naomipark/Desktop/jpl_internship/naomipark_mirsi/border_detection/dataset/train/" + str(i) + ".npy", params[0], params[1], params[2]])


def write(csvFile, row):
    writer = csv.writer(csvFile)
    writer.writerows([row])


if __name__ == '__main__':
    train_set()