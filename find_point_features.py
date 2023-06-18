import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('feature1.jpg')
# plt.figure(figsize=(8, 8))
# plt.imshow(img[:, :, ::-1])
# plt.show()

# # Harris corner detection
# img = cv.imread('feature1.jpg')

# blockSize = 2
# ksize = 3
# k = 0.04
# borderType = cv.BORDER_DEFAULT

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# harris_corner = cv.cornerHarris(
#     gray, blockSize=blockSize, ksize=ksize, k=k, borderType=borderType)
# # print(harris_corner.shape)

# corner_img = img.copy()
# thresh = 0.01*np.amax(harris_corner)
# corner_img[harris_corner > thresh] = (0, 0, 255)
# plt.figure(figsize=(8, 8))
# plt.imshow(corner_img[:, :, ::-1])
# plt.show()


# # Fast corner detection
# img = cv.imread('feature1.jpg')
# nfeatures = 500
# threshold = 20
# nonmaxSuppression = True
# type = cv.FAST_FEATURE_DETECTOR_TYPE_9_16
# fast = cv.FastFeatureDetector_create(
#     threshold=threshold, nonmaxSuppression=nonmaxSuppression, type=type)
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# fast_kps = fast.detect(gray, None)
# fast_kps_img = cv.drawKeypoints(img, keypoints=fast_kps, outImage=None)
# plt.figure(figsize=(8, 8))
# plt.imshow(fast_kps_img[:, :, ::-1])
# plt.show()


# # Oriented Rotated BRIEF
# img = cv.imread('feature1.jpg')
# nfeatures = 500
# scaleFactor = 1.2
# nlevels = 8
# edgeThreshold = 31
# firstLevel = 0
# WTA_K = 2
# scoreType = cv.ORB_HARRIS_SCORE
# fastThreshold = 20
# orb = cv.ORB_create(nfeatures=nfeatures,
#                     scaleFactor=scaleFactor,
#                     nlevels=nlevels,
#                     edgeThreshold=edgeThreshold,
#                     firstLevel=firstLevel,
#                     WTA_K=WTA_K,
#                     scoreType=scoreType,
#                     fastThreshold=fastThreshold
#                     )

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# orb_kps = orb.detect(gray, None)
# orb_kps_img = cv.drawKeypoints(img, keypoints=orb_kps, outImage=None)
# plt.figure(figsize=(8, 8))
# plt.imshow(orb_kps_img[:, :, ::-1])
# plt.show()


# SIFT feature detection
img = cv.imread('feature1.jpg')
nfeatures = 500
nOctaveLayers = 3
constrastThreshold = 0.04
edgeThreshold = 10
sigma = 1.6
sift = cv.xfeature2d.SIFT_create(nfeatures=nfeatures,
                                 nOctaveLayers=nOctaveLayers,
                                 constrastThreshold=constrastThreshold,
                                 edgeThreshold=edgeThreshold,
                                 sigma=sigma,
                                 )

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
sift_kps = sift.detect(gray, None)
sift_kps_img = cv.drawKeypoints(img, keypoints=sift_kps, outImage=None)
plt.figure(figsize=(8, 8))
plt.imshow(sift_kps_img[:, :, ::-1])
plt.show()
