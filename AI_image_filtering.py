import cv2
import numpy as np
import AI_function as func
import math
import time


def nine_different_GF_images(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    kernel_size = [5, 7, 13]
    sigma_value = [1, 5, 7]
    # make cases
    cases = [(i, j) for i in kernel_size for j in sigma_value]

    # in every cases
    res_image = np.array([])
    for ksize in kernel_size:
        tmp_images = np.array([])
        for sigma in sigma_value:

            Gaussian_2D = func.get_gaussian_filter_2d(ksize, sigma)
            # make filtered image
            filtered_image = func.cross_correlation_2d(image.copy(), Gaussian_2D)

            # write filter info on the image
            text = '%dx%d s=%d' % (ksize, ksize, sigma)
            font = cv2.FONT_HERSHEY_SIMPLEX
            filtered_image = cv2.putText(filtered_image, text, (10, 25), font, 0.6, (0, 0, 0), 1)

            if (tmp_images.size == 0):
                tmp_images = filtered_image
            else:
                tmp_images = np.concatenate((tmp_images, filtered_image), axis=1)


        if (res_image.size == 0):
            res_image = tmp_images
        else:
            res_image = np.concatenate((res_image, tmp_images), axis=0)

    return res_image


def differece_between_1D_and_2D_GF(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    size, sigma = 5, 1
    h_GF = func.get_gaussian_filter_1d(size, sigma)  # (1,5)
    v_GF = h_GF.reshape(size, 1)  # (5,1)
    # 1-Dimesion
    start = time.time()
    image1 = func.cross_correlation_1d(255 * func.cross_correlation_1d(image, v_GF), h_GF)
    print("gaussian 1D-filtering time: ", time.time() - start)

    # 2-Dimension
    start = time.time()
    image2 = func.cross_correlation_2d(image, func.get_gaussian_filter_2d(size, sigma))
    print("gaussian 2D-filtering time: ", time.time() - start)

    diff = abs(image1) - abs(image2)
    print("intensity sum: ", diff.sum())

    return diff

# Part 1 script

image_path = ['lenna.png', 'shapes.png']
kernel_1D = np.array([0,1,0])
kernel_2D = np.array([[0,0,0],
                      [0,1,0],
                      [0,0,0]])

# 1-2 : The Gaussian Filter

print('Gaussian Filter 1D')
print(func.get_gaussian_filter_1d(5,1))
print('Gaussian Filter 2D')
print(func.get_gaussian_filter_2d(5,1))

for path in image_path:
    # 1-1 : Image Filtering by Cross-Correlation
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    filtered_img = func.cross_correlation_1d(img, kernel_1D)  # 1D corr
    filtered_img = func.cross_correlation_2d(img, kernel_2D)  # 2D corr

    # 1-2 : The Gaussian Filter

    # save 9 images which are filtered by gaussian filters
    _9_images = nine_different_GF_images(path)

    cv2.imshow("9 images", _9_images)
    cv2.imwrite("./result/part_1_gaussian_filtered_" + str(image_path), 255 * _9_images)

    # differece between 1D and 2D Gaussian Filtering
    diff_map = differece_between_1D_and_2D_GF(path)

    cv2.imshow('difference map', diff_map)

    cv2.waitKey()
    cv2.destroyAllWindows()
    #


