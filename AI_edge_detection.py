import cv2
import numpy as np
import AI_function as func
import math
import time

def compute_image_gradient(image):
    # time recording
    start = time.time()

    # mag, dir 값 포함하는 3차원 np array 생성 - [2][y][x]
    output = np.zeros([2] + list(image.shape))

    # calculating gradient
    xdir_grad = func.sobel_filtering(255 * image, axis=0)
    ydir_grad = func.sobel_filtering(255 * image, axis=1)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            df_dx = xdir_grad[y, x]
            df_dy = ydir_grad[y, x]
            # direction 계산

            # 1사분면을 양수로.
            dir = math.atan2(-df_dy, df_dx) * (180 / math.pi)
            dir = 180 + dir if dir < 0 else dir

            # magnitude 계
            mag = math.pow((df_dx * df_dx) + (df_dy * df_dy), 1 / 2)
            # print("mag",mag)

            output[0][y, x] = mag
            output[1][y, x] = dir


    print("computing image gradient Time: ", time.time() - start)

    return output[0], output[1]


def non_maximum_suppression_dir(mag,dir):
    M, N = mag.shape
    # direction to array index pair
    dy = [[0, 0], [-1, 1], [-1, 1], [-1, 1]]
    dx = [[1, -1], [1, -1], [0, 0], [-1, 1]]
    output = mag.copy()

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            # quantize
            q = int((22.5 + dir[i, j]) / 45) % 4  # 0 ~ 3 values
            # suppression
            if (mag[i, j] <= mag[i + dy[q][0], j + dx[q][0]]) or (mag[i, j] <= mag[i + dy[q][1], j + dx[q][1]]):
                output[i, j] = 0

    return output


# Part 2 script

image_paths = ['lenna.png', 'shapes.png']
for path in image_paths:
    # read image
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # 2-1 Apply the Gaussian filtering to the input image
    image = func.cross_correlation_2d(image, func.get_gaussian_filter_2d(7, 1.5))

    # 2-2 Implement a function that returns the image gradient
    mag, dir = compute_image_gradient(image)

    cv2.imshow('image gradient', mag)
    cv2.imwrite("./result/part2_edge_raw_" + path, 255 * mag)

    # 2-3 Implement a function that performs Non-maximum Suppression (NMS)
    supressed_mag = non_maximum_suppression_dir(mag,dir)

    cv2.imshow("image suppressed", supressed_mag)
    cv2.imwrite("./result/part2_edge_sup_" + path, 255 * supressed_mag)

    cv2.waitKey()
    cv2.destroyAllWindows()


