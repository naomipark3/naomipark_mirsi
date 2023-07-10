import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import imageio
import cv2

im = fits.open('/Users/naomipark/Desktop/jpl_internship/naomipark_mirsi/data/wjup.00059.a.fits.gz') 
data = im[0].data


plt.imsave('ADMM_jup.jpg', data)


# Load the image in grayscale
gray_jpg = cv2.imread('/Users/naomipark/Desktop/jpl_internship/naomipark_mirsi/data/minus_edge_padding.jpg', cv2.IMREAD_GRAYSCALE)
#returns a numpy array

M = gray_jpg.shape[0] #rows
N = gray_jpg.shape[1] #cols

plt.imshow(gray_jpg, cmap='gray')
plt.show()
cv2.imwrite('grayscale_cleaned.jpg', gray_jpg)

def prox_absolute_value(z):
    out = 0
    if z >= 1:
        out = z-1
    if z <= -1:
        out = z+1
    return out

def update_D(lambda_down_k, lambda_right_k, y_k):
    d_down_kp1 = np.zeros((M-1, N))
    d_right_kp1 = np.zeros((M, N-1))

    for i in range(0, M-1): #need to define M, N
        for j in range(0, N):
            d_down_kp1[i,j] = prox_absolute_value(y_k[i+1, j] - y_k[i,j] - lambda_down_k[i,j]) #how do I initialize??

    for i in range(0, M):
        for j in range(0, N-1):
            d_right_kp1[i,j] = prox_absolute_value(y_k[i,j+1] - y_k[i,j] - lambda_right_k[i,j])
    
    return d_down_kp1, d_right_kp1

def update_Y(y, lambda_down_k, lambda_right_k, d_down_k, d_right_k, x, t):
    M, N = x.shape

    y_p1 = y.copy()

    for k in range(15):
        for i in range(M):
            for j in range(N):
                nb_neighbors = 0

                y_im1_j = 0 if i == 0 else y_p1[i-1, j]
                nb_neighbors += int(i > 0)

                y_ip1_j = 0 if i == M-1 else y_p1[i+1, j]
                nb_neighbors += int(i < M-1)

                y_i_jm1 = 0 if j == 0 else y_p1[i, j-1]
                nb_neighbors += int(j > 0)

                y_i_jp1 = 0 if j == N-1 else y_p1[i, j+1]
                nb_neighbors += int(j < N-1)

                dr_i_j = 0 if j == N-1 else d_right_k[i, j]
                dd_i_j = 0 if i == M-1 else d_down_k[i, j]
                dr_i_jm1 = 0 if j == 0 else d_right_k[i, j-1]
                dd_im1_j = 0 if i == 0 else d_down_k[i-1, j]

                lr_i_j = 0 if j == N-1 else lambda_right_k[i, j]
                ld_i_j = 0 if i == M-1 else lambda_down_k[i, j]
                lr_i_jm1 = 0 if j == 0 else lambda_right_k[i, j-1]
                ld_im1_j = 0 if i == 0 else lambda_down_k[i-1, j]

                tmp_y = y_im1_j + y_ip1_j + y_i_jm1 + y_i_jp1
                tmp_d = dr_i_jm1 - dr_i_j + dd_im1_j - dd_i_j

                tmp_1 = lr_i_jm1 - lr_i_j + ld_im1_j - ld_i_j
                tmp = t*x[i, j] + tmp_y + (tmp_1 + tmp_d)
                tmp = tmp/(t + nb_neighbors)
                y_p1[i, j] = tmp
    return y_p1


def update_lambda(lambda_down_k, lambda_right_k, d_down_kp1, d_right_kp1, y_kp1):
    lambda_down_kp1 = np.zeros((M-1, N))
    lambda_right_kp1 = np.zeros((M, N-1))
    for i in range(0, M-1):
        for j in range(0, N):
            lambda_down_kp1[i,j] = lambda_down_k[i,j] + d_down_kp1[i,j] - (y_kp1[i+1, j] - y_kp1[i,j])
    
    for i in range(0, M):
        for j in range(0,N-1):
            lambda_right_kp1[i,j] = lambda_right_k[i,j] + d_right_kp1[i,j] - (y_kp1[i,j+1] - y_kp1[i,j])
    return lambda_down_kp1, lambda_right_kp1


def ADMM(x, t, myepsilon):
    k=0 #initializes number of iterations
    y_k = x
    d_k = np.zeros((M,N))
    l_k = np.zeros((M,N))
    lambda_down_k = np.zeros((M-1,N))
    lambda_right_k = np.zeros((M,N-1))
    d_down_k = np.zeros((M-1, N))
    d_right_k = np.zeros((M,N-1))

    shall_continue = True

    while shall_continue == True:
        k = k+1

        d_down_kp1, d_right_kp1 = update_D(lambda_down_k, lambda_right_k, y_k)
        y_kp1 = update_Y(y_k, lambda_down_k, lambda_right_k, d_down_kp1, d_right_kp1, x, t)
        lambda_down_kp1, lambda_right_kp1 = update_lambda(lambda_down_k, lambda_right_k, d_down_kp1, d_right_kp1, y_kp1)

        residual = y_kp1 - y_k
        if np.linalg.norm(residual) <= myepsilon:
            shall_continue = False

        d_down_k = d_down_kp1
        d_right_k = d_right_kp1

        y_k = y_kp1
        
        lambda_down_k = lambda_down_kp1
        lambda_right_k = lambda_right_kp1

    niter = k
    return y_k, niter


x = gray_jpg / 255.0 #not sure if this needs to be scaled/if scaling makes a difference
myepsilon = 10 ** (-2)

for t in range(5, 45, 5):
    y_ki, niter = ADMM(x,t, myepsilon)
    if (t == 20):
        fig2 = plt.figure(2)
        plt.imshow(y_ki)
        plt.title("Denoised Image")
        plt.show()
        print("showing")
        exit()






