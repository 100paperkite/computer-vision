import cv2
import numpy as np
from func import *
from compute_avg_reproj_error import compute_avg_reproj_error


def find_matrix_A(srcP, destP):
    # [x,y,_x,_y] form
    points = np.hstack((srcP, destP))
    return np.array([[x * _x, x * _y, x, y * _x, y * _y, y, _x, _y, 1] for x, y, _x, _y in points])


def compute_F_raw(M):
    srcP, destP = M[:, 0:2], M[:, 2:4]

    # find the SVD of A
    A = find_matrix_A(srcP, destP)
    _, _, V = np.linalg.svd(A)

    # the least singular value
    F = V[-1, :]
    F = np.reshape(F, (3, 3))
    return F


def compute_F_norm(M):
    P1, P2 = M[:, 0:2], M[:, 2:4]

    # get normalizing matrixes
    N1 = get_normalize_matrix(P1)
    N2 = get_normalize_matrix(P2)

    # get transformed coordinates
    NormP1 = transform_coord(N1, P1)
    NormP2 = transform_coord(N2, P2)


    #  Find the SVD of A
    A = find_matrix_A(NormP1, NormP2)
    # A = np.dot(A.transpose(),A)

    _, _, V = np.linalg.svd(np.asarray(A))  # SVD


    #  Entries of F are the elements of column of V \
    #  corresponding to the least singular value
    F = np.array(V[-1, :]).reshape(3,3)
    # Enforce rank 2 constraint on F
    U, S, V = np.linalg.svd(F)
    S[2] = 0

    F = np.dot(U, np.dot(np.diag(S), V))

    # de-normalization
    return np.dot(np.dot(N1.transpose(), F), N2)


def compute_F_mine(M):
    P1, P2 = M[:, :2], M[:, 2:4]
    iteration = 1600
    max_matched = 0
    inliers = []  # 매칭되는 inlier 좌표의 index들

    P1 = np.hstack((P1, np.ones((len(M), 1))))
    P2 = np.hstack((P2, np.ones((len(M), 1))))
    Error = 100
    np.random.seed(0)
    for iters in range(iteration):
        # pick random 4 points
        rand4Idx = np.random.choice(len(M), 4, replace=False, p=None)
        # sampled 4 entries
        sample_P1, sample_P2 = P1[rand4Idx], P2[rand4Idx]

        _F = compute_F_norm(np.hstack((sample_P1[:, :2], sample_P2[:, :2])))

        # 0.14 : 1.26, 6.18, 3.81 / 0.15 : 1.26, 6.15, 3.61
        err, _inliers = my_compute_avg_reproj_error(M, _F, 0.15)
        if len(_inliers) < 4:
            continue
        _Error = compute_avg_reproj_error(M, compute_F_norm(np.hstack((P1[_inliers, :2], P2[_inliers, :2]))))

        # 제일 많이 매칭된다면, 저장
        if _Error < Error:
            inliers = _inliers
            max_matched = len(_inliers)
            F = _F
            Error = _Error

    # print(inliers)
    # 가장 잘 매칭된 것들끼리 homography를 다시 구해서 반환
    if (len(inliers) <= 4):
        return compute_F_norm(M)

    return compute_F_norm(np.hstack((P1[inliers, :2], P2[inliers, :2])))


def two_points_from_line(line, length):
    # ax+by+c
    pt1 = (length, ((length * line[0] + line[2]) / -line[1]))
    pt2 = (0, (line[2]) / -line[1])

    return pt1, pt2


def draw_3epipolar_lines(_image, F, _from, _to):
    image = _image.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
    for i in range(len(_from)):
        # to homogeneous
        points = np.append(_from[i], 1).astype(int)
        line = np.dot(F, np.reshape(points, (3, 1)))
        pt1, pt2 = two_points_from_line(line, image.shape[1])

        cv2.line(image, pt1, pt2, colors[i], 2)
        cv2.circle(image, tuple(_to[i].astype(int)), 4, colors[i], 2)

    return image


# numpy 출력옵션 변경
np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.4f}".format(x)})

images = [['temple1.png', 'temple2.png'],
          ['house1.jpg', 'house2.jpg'],
          ['library1.jpg', 'library2.jpg']]

for path1, path2 in images:
    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)

    # 1-1 Fundamental matrix computation
    M = np.loadtxt(path1.split('.')[0][:-1] + '_matches.txt')
    print("\nAverage Reprojection Errors (%s and %s)" % (path1, path2))
    print("   Raw =", compute_avg_reproj_error(M, compute_F_raw(M)))
    print("   Norm =", compute_avg_reproj_error(M, compute_F_norm(M)))
    print("   Mine =", compute_avg_reproj_error(M, compute_F_mine(M)))
    Mine = compute_F_norm(M)
    # 1-2 Visualization of epipolar lines
    while True:
        np.random.seed()
        # pick 3 random points
        rand3idx = np.random.choice(M.shape[0], 3, replace=False, p=None)
        rand3pts = M[rand3idx]
        result = np.hstack((draw_3epipolar_lines(image1, Mine.transpose(), rand3pts[:, 2:4], rand3pts[:, :2]),
                            draw_3epipolar_lines(image2, Mine, rand3pts[:, :2], rand3pts[:, 2:4]),))
        cv2.imshow('result', result)
        key = cv2.waitKey()
        if key == ord('p'):
            continue
        if key == ord('q'):
            break
