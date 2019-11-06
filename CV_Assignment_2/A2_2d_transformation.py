import numpy as np
import cv2
import math
import time


def transition_matrix(dir, amount):
    dy, dx = dir
    return np.array([[1, 0, dx * amount],
                     [0, 1, dy * amount],
                     [0, 0, 1]])


def rotation_matrix(degree):
    cost = math.cos((math.pi / 180) * degree)
    sint = math.sin((math.pi / 180) * degree)
    return np.array([[cost, -sint, 0],
                     [sint, cost, 0],
                     [0, 0, 1]])


def filp_matrix(axis):
    if axis == 1:
        return np.array([[-1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])
    elif axis == 0:
        return np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, 1]])


def scaling_matrix(axis, percent):
    dx, dy = not axis, axis
    p = percent / 100

    return np.array([[1+dx*p, 0, 0],
                     [0, 1+dy*p, 0],
                     [0, 0, 1]])


def draw_axis(image):
    cv2.arrowedLine(image, (400, 800), (400, 0), 0, 1, tipLength=.01)
    cv2.arrowedLine(image, (0, 400), (800, 400), 0, 1, tipLength=.01)
    return


def get_transformed_image(img, M):
    # 이미지의 중심 찾기
    yhalf, xhalf = int(img.shape[0] / 2), int(img.shape[1] / 2)

    # 그림에서 실제 값이 있는 좌표의 인덱스 알아내기
    imgIndex = np.where(img < 255)
    yidxes, xidxes = -imgIndex[0]+yhalf, imgIndex[1]-xhalf


    result_img = np.full((801, 801), 255, dtype=float) # 배경이미지 생성

    # 실제 값의 좌표들을 행렬 M을 이용하여 transform 시키기.
    for i in range(len(yidxes)):
        # 확대 시 이미지 깨짐 방지 - pixel 0.1단위로 세분화
        for d in np.arange(0,1,0.15):
            # 바뀐 좌표 _x, _y
            _x, _y, _ = np.dot(M, np.array([[xidxes[i]+d], [yidxes[i]+d], [1]]))
            # 배경이미지의 중심이 이미지의 중심으로 맞추고, 실제로 매핑시키기.
            result_img[-int(_y)+400,int(_x)+400] = img[-yidxes[i]+yhalf, xidxes[i]+xhalf]

    draw_axis(result_img)
    return result_img


input_img = 'smile.png'
img = cv2.imread(input_img, cv2.IMREAD_GRAYSCALE)

I = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])
M = I

while True:
    cv2.imshow('result', get_transformed_image(img, M))
    keyboard = cv2.waitKeyEx(0)

    if keyboard == ord('a'):
        M = np.dot(transition_matrix((0, -1), 5), M)
    elif keyboard == ord('d'):
        M = np.dot(transition_matrix((0, 1), 5), M)
    elif keyboard == ord('w'):
        M = np.dot(transition_matrix((1, 0), 5), M)
    elif keyboard == ord('s'):
        M = np.dot(transition_matrix((-1, 0), 5), M)
    elif keyboard == ord('r'):
        M = np.dot(rotation_matrix(5), M)
    elif keyboard == ord('q'):
        M = np.dot(rotation_matrix(-5), M)
    elif keyboard == ord('f'):
        M = np.dot(filp_matrix(1), M)
    elif keyboard == ord('g'):
        M = np.dot(filp_matrix(0), M)
    elif keyboard == ord('x'):
        M = np.dot(scaling_matrix(0, -5), M)
    elif keyboard == ord('z'):
        M = np.dot(scaling_matrix(0, 5), M)
    elif keyboard == ord('y'):
        M = np.dot(scaling_matrix(1, -5), M)
    elif keyboard == ord('t'):
        M = np.dot(scaling_matrix(1, 5), M)
    elif keyboard == ord('m'):
        M = I
    elif keyboard == ord('p'):
        break

cv2.destroyAllWindows()

# while True:
#    keyboard = input()
