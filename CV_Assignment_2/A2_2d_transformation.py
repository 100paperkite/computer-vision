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
    Y, X = -imgIndex[0]+yhalf, imgIndex[1]-xhalf

    result_img = np.full((801, 801), 255, dtype=float) # 배경이미지 생성

    # 실제 값의 좌표들을 행렬 M을 이용하여 transform 시키기.
    x = list([0,0,0,0])
    y = list([0,0,0,0])
    d = 0.5
    for i in range(len(Y)):
        # 이미지 깨짐 방지 위해 픽셀값 -d~+d 범위 내 픽셀 모두 고려.
        # 이미지의 원점과 배경이미지의 원점을 맞추기 위해 +400 더해줌
        _x,_y, _ = np.dot(M, np.array([[X[i]], [Y[i]], [1]])) + 400 # 원래 매핑 되어야 하는 값
        [x[0]], [y[0]], _ = np.dot(M, np.array([[X[i]+d], [Y[i]+d], [1]])) +400
        [x[1]], [y[1]], _ = np.dot(M, np.array([[X[i]-d], [Y[i]-d], [1]])) +400
        [x[2]], [y[2]], _ = np.dot(M, np.array([[X[i]+d], [Y[i]-d], [1]])) +400
        [x[3]], [y[3]], _ = np.dot(M, np.array([[X[i]-d], [Y[i]+d], [1]])) +400

        x = sorted(x) # 최대 x, 최소 x 찾기 위해 sort
        y = sorted(y) # 최대 y, 최소 y 찾기 위해 sort

        # 범위 내의 픽셀값들에 전부 기존 값 매핑시키기. (y는 좌표상에서 -/+ 반대이기 때문에 -부호 처리해줌)
        result_img[int(-y[3]):int(-y[0]),int(x[0]):int(x[3])] = result_img[int(-_y),int(_x)] = img[-Y[i]+yhalf, X[i]+xhalf]

    draw_axis(result_img)

    return result_img/255


input_img = 'smile.png'
img = cv2.imread(input_img, cv2.IMREAD_GRAYSCALE)

I = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])
M = I

while True:
    cv2.imshow('result', get_transformed_image(img, M))
    keyboard = cv2.waitKey()

    if keyboard == ord('a'):
        M = np.dot(transition_matrix((0, -1), 5), M)
    elif keyboard == ord('d'):
        M = np.dot(transition_matrix((0, 1), 5), M)
    elif keyboard == ord('w'):
        M = np.dot(transition_matrix((1, 0), 5), M)
    elif keyboard == ord('s'):
        M = np.dot(transition_matrix((-1, 0), 5), M)
    elif keyboard == ord('R'):
        M = np.dot(rotation_matrix(-5), M)
    elif keyboard == ord('r'):
        M = np.dot(rotation_matrix(5), M)
    elif keyboard == ord('F'):
        M = np.dot(filp_matrix(0), M)
    elif keyboard == ord('f'):
        M = np.dot(filp_matrix(1), M)
    elif keyboard == ord('X'):
        M = np.dot(scaling_matrix(0, 5), M)
    elif keyboard == ord('x'):
        M = np.dot(scaling_matrix(0, -5), M)
    elif keyboard == ord('Y'):
        M = np.dot(scaling_matrix(1, 5), M)
    elif keyboard == ord('y'):
        M = np.dot(scaling_matrix(1, -5), M)
    elif keyboard == ord('H'):
        M = I
    elif keyboard == ord('Q'):
        break

cv2.destroyAllWindows()

