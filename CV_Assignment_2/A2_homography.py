import numpy as np
import cv2
import time

CONST_R = (1 << np.arange(8))[:,None]
# 2-1
def sum_hamming_distance(v1, v2):
    return np.count_nonzero((v1 & CONST_R) != (v2 & CONST_R))


def BF_match(des1, des2):
    size = len(des1)
    maxV = 8 * size

    matches = []
    for i in range(size):
        sumlist = [sum_hamming_distance(des1[i], cmp) for cmp in des2]
        minIdx = sumlist.index(min(sumlist))
        matches.append((i,minIdx,sumlist[minIdx]))

    return matches

# 2-2
def compute_homography(srcP,desP):
    return



# main script

# Read Images
desk = cv2.imread('cv_desk.png', cv2.IMREAD_GRAYSCALE)
cover = cv2.imread('cv_cover.jpg', cv2.IMREAD_GRAYSCALE)

# kp - 2차원 좌표,  des = descriptor
# Hamming distance는 numpy
orb = cv2.ORB_create()

kp1 = orb.detect(desk,None)
kp1, des1 = orb.compute(desk, kp1)
kp2 = orb.detect(cover, None)
kp2, des2 = orb.compute(cover, kp2)

start = time.time()
matches = BF_match(des1,des2)
matches = sorted(matches, key = lambda x:x[2])
res=[]
res= [cv2.DMatch() for i in range(10)]
for i in range(10):
    res[i].queryIdx, res[i].trainIdx, res[i].distance = matches[i]

result = cv2.drawMatches(desk,kp1,cover,kp2,res,None,flags=2)
print(time.time()-start)

cv2.imshow('result',result)
cv2.waitKey(0)



