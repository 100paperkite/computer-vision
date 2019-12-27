import numpy as np


# For Problem 1

# 좌표 변환
def transform_coord(M, pts):
    size = pts.shape[0]
    # to homogeneous
    ones = np.ones((size, 1))
    pts = np.hstack((pts, ones))  # [x,y,1] form

    # apply matrix
    result = np.array([np.dot(M, coord.reshape(3, 1)) for coord in pts])
    result = result.reshape(size, 3)

    for coord in result:
        if coord[2] != 0:
            coord /= coord[2]

    return result[:, :2]


# 정규화 matrix 반환
def get_normalize_matrix(P):
    # 1. mean subtraction - move to the center to origin (0, 0)
    mean = np.mean(P, axis=0).astype(int)
    # 2. scaling to -1 ~ 1
    min = np.min(P - mean, axis=0)
    ptp = np.ptp(P - mean, axis=0)

    M = np.array([[1, 0, -mean[0]],
                  [0, 1, -mean[1]],
                  [0, 0, 1]])
    Scale = np.array([[1 / ptp[0], 0, -min[0] / ptp[0]],
                      [0, 1 / ptp[1], -min[1] / ptp[1]],
                      [0, 0, 1]])
    _S = np.array([[2, 0, -1],
                   [0, 2, -1],
                   [0, 0, 1]])

    return np.dot(np.dot(_S, Scale), M)


# normalization test
def my_compute_avg_reproj_error(_M, _F, th):
    N = _M.shape[0]

    X = np.c_[_M[:, 0:2], np.ones((N, 1))].transpose()
    L = np.matmul(_F, X).transpose()
    norms = np.sqrt(L[:, 0] ** 2 + L[:, 1] ** 2)
    L = np.divide(L, np.kron(np.ones((3, 1)), norms).transpose())
    L = (np.multiply(L, np.c_[_M[:, 2:4], np.ones((N, 1))])).sum(axis=1)
    error1 = (np.fabs(L)) / (N * 2)

    X = np.c_[_M[:, 2:4], np.ones((N, 1))].transpose()
    L = np.matmul(_F.transpose(), X).transpose()
    norms = np.sqrt(L[:, 0] ** 2 + L[:, 1] ** 2)
    L = np.divide(L, np.kron(np.ones((3, 1)), norms).transpose())
    L = (np.multiply(L, np.c_[_M[:, 0:2], np.ones((N, 1))])).sum(axis=1)
    error2 = (np.fabs(L)) / (N * 2)

    inliers = [i for i in range(N) if error1[i] + error2[i] <= th]

    return (error1.sum() + error2.sum()) / (N * 2), inliers


# For Problem 2

# L2 distance
def compute_L2(point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))


# return group label
def get_label(keypoints, new_centroids):
    MIN_DIS = compute_L2(keypoints, new_centroids[0])
    result = 0
    for label in range(1, len(new_centroids)):
        dis = compute_L2(keypoints, new_centroids[label])
        if dis < MIN_DIS:
            MIN_DIS = dis
            result = label

    return result


# init centroids
def init_centroids(points):
    # k-means++ approach
    centroids = []
    k = 8

    first = np.random.choice(len(points), 1, replace=False, p=None)

    centroids.append(points[first])
    for iter in range(k - 1):
        print("k++: ", iter + 1)
        centroids.append(k_plus_algorithm(centroids, points))
    return np.vstack(np.array(centroids))


def iterate_k_means(data_points, centroids, total_iteration):
    for iteration in range(total_iteration):
        print('iter:', iteration, end="  ")
        print('centroids:', np.sum(centroids))

        new_centroids = np.zeros((8,128))
        count = np.zeros(8)
        for p in range(len(data_points)):
            distance = {}
            for c in range(len(centroids)):
                # point와 centroid k개의 길이 각각 계산해서 dictionary로 저장
                distance[c] = compute_L2(data_points[p], centroids[c])
            # 최소 거리를 갖는 centroid로 label 붙임
            minIndex = min(distance, key=distance.get)
            count[minIndex] += 1
            # centroid 업데이트 - 평균냄
            new_centroids[minIndex] += data_points[p]

        for i in range(8):
            if count[i] == 0 : new_centroids = centroids[i]
            new_centroids[i] = new_centroids[i]/count[i]

        centroids = new_centroids

    return centroids


def k_plus_algorithm(metrics_points, points):
    distance = []
    for i in range(len(points)):
        d = 1000000000  # MAX
        for j in range(len(metrics_points)):
            TEMP_DIS = compute_L2(points[i], metrics_points[j])
            d = min(d, TEMP_DIS)
        distance.append(d)

    distance = np.array(distance)
    new_centroid = points[np.argmax(distance)]

    return new_centroid
