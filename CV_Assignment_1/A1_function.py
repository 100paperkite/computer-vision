import cv2
import numpy as np
import math
import time


#### For Assignment 1 ####

def image_padding(image, V, H):
    img = image.copy()

    # horizontal
    if H != 0:
        edge_left = np.zeros((img.shape[0], H))
        edge_right = np.zeros((img.shape[0], H))
        for y in range(img.shape[0]):
            edge_left[y].fill(img[y][0])
            edge_right[y].fill(img[y][-1])
        img = np.concatenate((np.concatenate((edge_left, img), axis=1), edge_right), axis=1)

    # vertical
    if V != 0:
        edge_up = img[0, :].copy()
        edge_down = img[-1, :].copy()
        for i in range(V - 1):
            edge_up = np.vstack((edge_up, img[0, :].copy()))
            edge_down = np.vstack((edge_down, img[-1, :].copy()))
        img = np.vstack((np.vstack((edge_up, img)), edge_down))

    return img


def cross_correlation_1d(img, kernel):
    output = np.zeros(img.shape)
    pad_size = int(kernel.shape[0] / 2)

    r, c = output.shape
    # horizontal
    if kernel.ndim == 1:
        img = image_padding(img, 0, pad_size)
        for y in range(r):
            for x in range(c):
                output[y, x] = (kernel * img[y, x: x + kernel.shape[0]]).sum()
    # vertical
    else:
        img = image_padding(img, pad_size, 0)
        _kernel = np.squeeze(kernel)
        for y in range(r):
            for x in range(c):
                output[y, x] = (_kernel * img[y: y + kernel.shape[0], x]).sum()

    output /= 255.0
    return output


def cross_correlation_2d(img, kernel):
    output = np.zeros(img.shape)
    v_pad_size, h_pad_size = int((kernel.shape[0])/2), int(kernel.shape[1]/2)
    img = image_padding(img, v_pad_size, h_pad_size)

    r,c  = output.shape

    for y in range(r):
        for x in range(c):
            output[y,x] = (kernel * img[y:y+kernel.shape[0],x:x+kernel.shape[1]]).sum()

    output /= 255.0

    return output


def gaussian_function(x, sigma):
    return (1 / (sigma*math.sqrt(2 * math.pi))) * pow(math.e, -(x * x) / (2 * sigma * sigma))


def get_gaussian_filter_1d(size, sigma):
    # mean = 0
    length = int(size / 2)
    output = np.array([gaussian_function(x, sigma) for x in range(-length, length + 1)])
    # 가우시안의 합은 1
    output /= sum(output)
    return output


def get_gaussian_filter_2d(size, sigma):
    # Gaussian은 separable
    row_1d = get_gaussian_filter_1d(size, sigma)
    col_1d = row_1d.reshape(size, 1)

    # nx1 * 1xn = nxn 가우시안 필터
    output = np.outer(col_1d, row_1d)
    output/= output.sum()

    return output


#### For Assignment 2 ####

def sobel_filtering(image, axis):
    blurring = np.array([1, 2, 1])
    derivative_1D = np.array([-1, 0, 1])
    if (axis == 0):
        return cross_correlation_2d(image, np.outer(blurring.reshape(3, 1),derivative_1D))
    else:
        return cross_correlation_2d(image, np.outer(derivative_1D.reshape(3, 1), blurring))




