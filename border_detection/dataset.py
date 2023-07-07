import numpy as np
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
    img += random.uniform(0.01, 1) * noise * np.random.rand(*img.shape)
    return (row, col, rad), img

def create_yolo_annotation(params, img_shape):
    row, col, rad = params
    img_size = img_shape[0]  # Assuming the image is square
    x_center, y_center, width, height = (col / img_size, row / img_size, (2 * rad) / img_size, (2 * rad) / img_size)
    return f"0 {x_center} {y_center} {width} {height}\n"  # assuming the class for circles is "0"

from skimage.io import imsave

#...

def dataset():
    number_of_images = 10088
    level_of_noise = 3.5
    # Set the split percentage for the training set
    train_split = 0.8  # 80% for training
    train_size = int(number_of_images * train_split)

    with open("dataset.csv", 'w', newline='') as outFile,\
         open("data/train.txt", 'w') as trainFile,\
         open("data/valid.txt", 'w') as validFile:
        header = ['NAME', 'ROW', 'COL', 'RAD', 'SET']
        write(outFile, header)
        for i in range(number_of_images):
            params, img = noisy_circle(200, 100, level_of_noise)
            img = (img - np.min(img)) / (np.max(img) - np.min(img))  # normalize image to [0,1]
            img_path = f"/Users/naomipark/Desktop/jpl_internship/naomipark_mirsi/border_detection/dataset/"
            if i < train_size:
                set_type = 'train'
                trainFile.write(img_path + set_type + '/' + str(i) + ".png\n")
            else:
                set_type = 'val'
                validFile.write(img_path + set_type + '/' + str(i) + ".png\n")
            imsave(img_path + set_type + '/' + str(i) + ".png", img)
            write(outFile, [img_path + set_type + '/' + str(i) + ".png", params[0], params[1], params[2], set_type])
            # Create YOLO annotation file
            with open(img_path + set_type + '/' + str(i) + ".txt", 'w') as file:
                file.write(create_yolo_annotation(params, img.shape))

def write(csvFile, row):
    writer = csv.writer(csvFile)
    writer.writerows([row])

if __name__ == '__main__':
    dataset()

