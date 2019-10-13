import cv2
import numpy as np
import math
import time


def image_padding(img, V, H):
    if V == 0 and H == 0:
        return img

    # 1D-horizontal
    if V == 0 and H != 0:
        edge_left = np.zeros((img.shape[0], H))
        edge_right = np.zeros((img.shape[0], H))
        for y in range(img.shape[0]):
            edge_left[y].fill(img[y][0])
            edge_right[y].fill(img[y][-1])
        return np.concatenate((np.concatenate((edge_left, img), axis=1), edge_right), axis=1)

    # 1D-vertical
    elif V != 0 and H == 0:
        edge_up = img[0, :].copy()
        edge_down = img[-1, :].copy()
        for i in range(V - 1):
            edge_up = np.vstack((edge_up, img[0, :].copy()))
            edge_down = np.vstack((edge_down, img[-1, :].copy()))
        return np.vstack((np.vstack((edge_up, img)), edge_down))
    # 2D padding
    else:
        h_padded = image_padding(img, 0, H)
        v_h_padded = image_padding(h_padded, V, 0)
        return v_h_padded


def cross_correlation_1d(img, kernel):
    output = np.zeros(img.shape)
    pad_size = int(kernel.shape[0] / 2)

    # horizontal
    if kernel.ndim == 1:
        img = image_padding(img, 0, pad_size)
        for y in range(output.shape[0]):
            for x in range(output.shape[1]):
                for dx in range(kernel.shape[0]):
                    output[y, x] += kernel[dx] * img[y, x + dx]
                output[y, x] /= 255.0
    # vertical
    else:
        img = image_padding(img, pad_size, 0)
        for y in range(output.shape[0]):
            for x in range(output.shape[1]):
                for dy in range(kernel.shape[0]):
                    output[y, x] += kernel[dy][0] * img[y + dy, x]
                output[y, x] /= 255.0

    return output


def cross_correlation_2d(img, kernel):
    output = np.zeros(img.shape)
    v_pad_size = int(kernel.shape[0] / 2)
    h_pad_size = int(kernel.shape[1] / 2)

    img = image_padding(img, v_pad_size, h_pad_size)

    kernel_idx = [(i, j) for i in range(kernel.shape[0]) for j in range(kernel.shape[1])]
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            for (dy, dx) in kernel_idx:
                output[y, x] += kernel[dy, dx] * img[y + dy, x + dx]
            output[y, x] /= 255.0

    return output


def gaussian_function(x, sigma):
    return (1 / pow(2 * math.pi, 1 / 2)) * pow(math.e, -(x * x) / (2 * sigma * sigma))


def get_gaussian_filter_1d(size, sigma):
    # mean = 0
    len = int(size / 2)
    output = np.array([gaussian_function(x, sigma) for x in range(-len, len + 1)])
    # 가우시안의 합은 1
    output = output * (1 / sum(output))

    return output


def get_gaussian_filter_2d(size, sigma):
    # Gaussian은 separable
    row_1d = get_gaussian_filter_1d(size, sigma)
    col_1d = row_1d.reshape(size, 1)

    # nx1 * 1xn = nxn 가우시안 필터
    res = np.outer(col_1d, row_1d)
    # print("* 합은 항상 1 :",sum(sum(res)))
    # print(res)

    return res


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

            Gaussian_2D = get_gaussian_filter_2d(ksize, sigma)
            # make filtered image
            filtered_image = cross_correlation_2d(image.copy(), Gaussian_2D)

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
    h_GF = get_gaussian_filter_1d(size, 1)  # (1,5)
    v_GF = h_GF.reshape(size, 1)  # (5,1)

    # ** checking for Cross-Correlation **
    # h_F = np.array([0,0,1,0,0])
    # v_F = np.array([[0],[0],[1],[0],[0]])
    # hv_F = np.array([[0,0,0],[0,1,0],[0,0,0]])

    # 1-Dimesion
    start = time.time()
    image1 = cross_correlation_1d(255*cross_correlation_1d(image, h_GF),v_GF)
    print("1D-time: ", time.time() - start)
    #cv2.imshow('image1',image1)

    # 2-Dimension
    start = time.time()
    image2 = cross_correlation_2d(image, get_gaussian_filter_2d(size,1))
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
