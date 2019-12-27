import sys
import numpy as np
import struct
import datetime
from func import *
import os

np.random.seed(1)

# # SIFT Descriptor random sampling
# desc = []
# for i in range(1000):
#     number = 100000 + i
#     with open('./sift/sift' + str(number), 'rb') as f:
#         while True:
#             for j in range(128):
#                 b = f.read(1)
#                 if not b:
#                     break
#                 desc.append(int(ord(b)))
#
#             if not b:
#                 break
#
# row = len(desc) // 128
# randidx = np.random.choice(row, 100000, replace=False, p=None)
# reshaped = np.reshape(desc, (row, 128))
# reshaped = reshaped[randidx]
# np.save("./100000_desc", reshaped) # sift feature 10만개만 random sampling


# ## k means clustering
# print("k-means clustering")
# points = np.load('100000_desc.npy')
# centroids = init_centroids(points)
# total_iteration = 30
# new_centroids = iterate_k_means(points[:, :], centroids, total_iteration)
# np.save('k++_centroids8', new_centroids)


# VLAD image classification
print("start classification")
new_centroids = np.vstack(np.load('k++_centroids8.npy')) # read centroids
result_desc = np.zeros((1000, 1024))
desc = []
for i in range(1000):
    if i % 100 == 0: print(str(i) + 'th') # 확인용
    number = 100000 + i

    # 전체 keypoints들에 대해 vlad
    with open('./sift/sift' + str(number), 'rb') as f:
        cnt = np.zeros(8)
        while True:
            keypoints = np.zeros(128)
            for j in range(128):
                b = f.read(1)
                if not b:
                    break
                keypoints[j] = (int(ord(b)))
            if not b:
                break

            label = get_label(keypoints, new_centroids)  # get group label
            cnt[label] += 1  # count the number of group
            _label = label * 128
            result_desc[i, _label:_label + 128] += keypoints - new_centroids[label]  # add vector

        # square-rooting normalization
        result_desc[i] = np.sign(result_desc[i]) * np.sqrt(np.abs(result_desc[i]))

        # L2 normalization
        result_desc[i] = result_desc[i] / np.sqrt(np.dot(result_desc[i], result_desc[i]))

# write binary file
with open('A3_2016312607.des', 'wb') as write_binary:
    write_binary.write(struct.pack('ii', 1000, 1024))
    write_binary.write(result_desc.astype('float32').tobytes())
