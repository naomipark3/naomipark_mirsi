import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Open the PGM image
pgm_image = Image.open("/Users/naomipark/Desktop/jpl_internship/naomipark_mirsi/data/cleaned_ADMM.pgm")

color_img = pgm_image.convert('RGB')

color_img.save('output.jpg')

