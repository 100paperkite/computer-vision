import cv2
import numpy as np
import AI_function as func
import math
import time




def save_9_different_GF_images(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    kernel_size = [5, 11, 17]
    sigma_value = [1, 5, 11]
    # make cases
    cases = [(i, j) for i in kernel_size for j in sigma_value]

    cnt = 0

    # in every cases
    res_image = np.array([])
    for ksize in kernel_size:
        tmp_images = np.array([])
        for sigma in sigma_value:

            # for debugging
            print("%dth on going ..." % cnt)
            cnt += 1
            #

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

    # image verification
    cv2.imshow("result images", res_image)
    # saving image
    cv2.imwrite("./result/part_1_gaussian_filtered_" + image_path, 255 * res_image)


def differece_between_1D_and_2D_GF(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    size = 5
    h_GF = func.get_gaussian_filter_1d(size, 1)  # (1,5)
    v_GF = h_GF.reshape(size, 1)  # (5,1)

    # ** checking for Cross-Correlation **
    # h_F = np.array([0,0,1,0,0])
    # v_F = np.array([[0],[0],[1],[0],[0]])
    # hv_F = np.array([[0,0,0],[0,1,0],[0,0,0]])

    # 1-Dimesion
    start = time.time()
    image1 = func.cross_correlation_1d(255*func.cross_correlation_1d(image, h_GF),v_GF)
    print("1D-time: ", time.time() - start)
    #cv2.imshow('image1',image1)

    # 2-Dimension
    start = time.time()
    image2 = func.cross_correlation_2d(image, func.get_gaussian_filter_2d(size,1))
    print("2D-time: ", time.time() - start)
    #cv2.imshow('image2', image2)

    for y in range(image.shape[0]):
        intensity_sum = 0
        for x in range(image.shape[1]):
            diff = abs(image1[y, x] - image2[y, x])
            print("%.1f " % diff, end=' ')
            intensity_sum +=diff
        print("")
    print("intensity sum: ",intensity_sum)


# main script

# Problem #1 : Image Filtering by Cross-Correlation

# Problem #2 : The Gaussian Filter
# print("\n***** Gaussian 1-Dimension Filter *****\n")
# print(get_gaussian_filter_1d(5, 1))
# print("\n***** Gaussian 2-Dimension Filter *****\n")
# print(get_gaussian_filter_2d(5, 1))
# save_9_different_GF_images('lenna.png')
# cv2.waitKey()
# cv2.destroyAllWindows()
#
# differece_between_1D_and_2D_GF("lenna.png")
