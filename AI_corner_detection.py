import AI_function as func
import cv2
import numpy as np
import time

def compute_corner_response(image_path):
    start = time.time()
    # image read
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Gaussian filtering
    image = func.cross_correlation_2d(gray_image, func.get_gaussian_filter_2d(7, 1.5))

    Ix = func.sobel_filtering(image,axis=0)
    Iy = func.sobel_filtering(image,axis=1)

    wsize = 5   # window size = 5
    Kappa = 0.04

    M = np.zeros((2,2))
    response = np.zeros(image.shape)
    for i in range(image.shape[0]-wsize):
        for j in range(image.shape[1]-wsize):
            # compute second moment matrix
            window_Ix = Ix[i:i+wsize,j:j+wsize]
            window_Iy = Iy[i:i+wsize,j:j+wsize]
            a = M[0,0] = (window_Ix*window_Ix).sum()    # Ix^2
            b = M[0,1] = (window_Ix*window_Iy).sum()    # IxIy
            c = M[1,0] = M[0,1]                         # IxIy
            d = M[1,1] = (window_Iy*window_Iy).sum()    # Iy^2

            # response function
            # R = det(M) - K*trace^2(M)
            Res = (a*d-b*c) - Kappa*pow(a+d,2)
            if(Res<0) : Res = 0

            # because window size is 5
            response[i+2,j+2] = Res

    response = response / response.max()

    print("Computing time of corner response",time.time()-start)


    # corner raw image
    cv2.imshow('corner detection', response)
    cv2.imwrite('./result/part_3_corner_raw_'+image_path,255*response)

    # corner bin image
    bin_img = corner_bin(gray_image,response)

    cv2.imshow('corner bin', bin_img)
    cv2.imwrite('./result/part_3_corner_bin_'+image_path,bin_img)

    cv2.waitKey()

    return response


def corner_bin(image, response_img):
    rgb = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    threshold = 0.1
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            if response_img[i, j]>threshold:
                rgb[i,j,1] = 255
    return rgb

def non_maximum_suppression_win(R,winSize):
    start = time.time()
    output = R.copy()
    # suppress if < threshold
    output[np.where(output<0.1)] = 0

    # suppress if is not maximum
    r,c = output.shape
    d = int(winSize/2)
    for i in range(r-winSize):
        for j in range(c-winSize):
            window = output[i:i+winSize,j:j+winSize]
            if window.max()!=output[i+d,j+d]:
                output[i+d,j+d] = 0

    print("Computing time of NMS response",time.time()-start)

    return output

def draw_circle_at_point(img,suppressed_R):
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    output = rgb.copy()
    for i in range(suppressed_R.shape[0]):
        for j in range(suppressed_R.shape[1]):
            if suppressed_R[i,j]>0:
                cv2.circle(output,(j,i),4,(0,255,0),2)

    return output



R = compute_corner_response('lenna.png')

suppressed_R = non_maximum_suppression_win(R,11)
gray_image = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('R',R)
cv2.imshow('NMS R',suppressed_R)
cv2.imshow('circled R',draw_circle_at_point(gray_image,suppressed_R))
cv2.waitKey()

