import numpy as np
import cv2
import time

CONST_R = (1 << np.arange(8))[:, None]

#
def wrap_image(base,addimg):
    result = base.copy()
    for i in range(base.shape[0]):
        for j in range(base.shape[1]):
            if addimg[i, j] != 0:
                result[i, j] = addimg[i, j]
    return result
#
def get_points(matches,kp1,kp2,num):
    matches = np.array(matches[:num])  # num 개수만큼 뽑기
    kp1 = np.array(kp1)
    kp2 = np.array(kp2)

    idx1 = matches[:, 0]
    idx2 = matches[:, 1]

    p1 = np.array([np.array(kp.pt) for kp in kp1[idx1]])
    p2 = np.array([np.array(kp.pt) for kp in kp2[idx2]])

    return p1,p2

# 2-1
def hamming_distance(v1, v2):
    return np.count_nonzero((v1 & CONST_R) != (v2 & CONST_R))


def BF_match(des1, des2):
    matches = []
    for i in range(len(des1)):
        h_dist = [hamming_distance(des1[i], cmp) for cmp in des2]
        minIdx = h_dist.index(min(h_dist))
        matches.append([i, minIdx, h_dist[minIdx]])

    return matches


def toDMatchList(matches):
    res = []
    res = [cv2.DMatch() for i in range(len(matches))]
    for i in range(len(matches)):
        res[i].queryIdx, res[i].trainIdx, res[i].distance = matches[i]
    return res


# 2-2

# srcP, desP : N x 2 matrices - N(# of matched points, and location in img)
# return value : 3 x 3 transformed matrix

def transform_coord(M, pts):
    size = pts.shape[0]
    # to homogeneous
    ones = np.ones((size, 1))
    pts = np.hstack((pts, ones))  # [x,y,1] form

    # apply matrix
    result = np.array([np.dot(M, coord.reshape(3, 1)) for coord in pts])
    result = result.reshape(size, 3)

    for coord in result:
        if coord[2]!=0:
            coord/=coord[2]

    return result[:, :2]


def get_normalize_matrix(P):
    m = np.mean(P, axis=0)  # 1. mean subtraction
    p = np.array([P[:, 0] - m[0], P[:, 1] - m[1]])
    min, max = np.min(p), np.max(p)  # 2. scaling
    s = max - min

    meanMat = np.array([[1, 0, -m[0]], [0, 1, -m[1]], [0, 0, 1]])

    s1 = np.array([[1, 0, -min], [0, 1, -min], [0, 0, 1]])
    scaleMat = np.dot(np.array([[1 / s, 0, 0], [0, 1 / s, 0], [0, 0, 1]]), s1)

    N = np.dot(scaleMat, meanMat)
    return N


def find_matrix_A(srcP, destP):
    size = srcP.shape[0]
    # [x,y,_x,_y] form
    points = np.hstack((srcP, destP))
    A = np.array([[[x, y, 1, 0, 0, 0, -x * _x, -y * _x, -_x],
                   [0, 0, 0, x, y, 1, -x * _y, -y * _y, -_y]] for x, y, _x, _y in points])

    return np.reshape(A, (size * 2, 9))


def compute_homography(srcP, destP):
    # get normalizing matrixes
    Ts = get_normalize_matrix(srcP)
    Td = get_normalize_matrix(destP)

    # get transformed coordinates
    normS = transform_coord(Ts, srcP)
    normD = transform_coord(Td, destP)

    # find matrix A, and h
    A = find_matrix_A(normS, normD)
    U, s, Vh = np.linalg.svd(np.asarray(A))  # SVD

    h = Vh[-1, :] / Vh[-1, -1]
    h = np.reshape(h, (3, 3))

    # back to before normalization
    Td_ = np.linalg.inv(Td)
    Tmat = np.dot(np.dot(Td_, h), Ts)

    return Tmat


def compute_homography_ransac(srcP, destP, th):
    iteration = 2500
    max_matched = 0

    inliers = []
    for iters in range(iteration):
        rand4Idx = np.random.choice(len(srcP), 4, replace=False, p=None)

        # sampled 4 entries
        sample_srcP, sample_destP = srcP[rand4Idx], destP[rand4Idx]

        _H = compute_homography(sample_srcP, sample_destP)
        test_destP = transform_coord(_H,srcP)

        _inliers = []
        for i in range(len(test_destP)):
            if abs(test_destP[i][0]-destP[i][0])<=th and abs(test_destP[i][1]-destP[i][1])<=th:
                _inliers.append(i)

        if len(_inliers) > max_matched:
            inliers=_inliers
            max_matched = len(_inliers)

    return compute_homography(srcP[inliers],destP[inliers])


# main script

# Read Images
desk = cv2.imread('cv_desk.png', cv2.IMREAD_GRAYSCALE)
cover = cv2.imread('cv_cover.jpg', cv2.IMREAD_GRAYSCALE)

# kp - 2차원 좌표,  des = descriptor
orb = cv2.ORB_create()
kp1 = orb.detect(desk, None)
kp1, des1 = orb.compute(desk, kp1)
kp2 = orb.detect(cover, None)
kp2, des2 = orb.compute(cover, kp2)

# 2-1 Feature detection, description, and matching

matches = BF_match(des1, des2)
matches = sorted(matches, key=lambda x: x[2])  # distance 정렬
feature_matched = cv2.drawMatches(desk, kp1, cover, kp2, toDMatchList(matches[:20]), None, flags=2)

# cv2.imshow('feature matching',feature_matched)
# cv2.waitKey(0)


# 2-2 Computing homography with normalization
deskP, coverP = get_points(matches,kp1,kp2,18)

T = compute_homography(coverP, deskP)
R,mask = cv2.findHomography(coverP,deskP,cv2.RANSAC)

transformed_img = cv2.warpPerspective(cover,T, (desk.shape[1],desk.shape[0]))

cv2.imshow('homography with normalization', wrap_image(desk,transformed_img))
cv2.waitKey(0)

# Computing homography with RANSAC
deskP, coverP = get_points(matches,kp1,kp2,150)
ransac = compute_homography_ransac(coverP,deskP,2.5)

result = desk.copy()
ransac_img = cv2.warpPerspective(cover, ransac, (desk.shape[1],desk.shape[0]))

cv2.imshow('ransac', wrap_image(desk,ransac_img))
cv2.waitKey(0)