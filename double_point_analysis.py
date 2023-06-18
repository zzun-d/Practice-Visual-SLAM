import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# img1 = cv.imread('feature1.jpg')
# img2 = cv.imread('feature2.jpg')

# orb = cv.ORB_create()

# gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
# gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# orb_kps1 = orb.detect(gray1, None)
# orb_kps2 = orb.detect(gray2, None)

# orb_kps_img1 = cv.drawKeypoints(
#     img1, keypoints=orb_kps1, outImage=None, color=(0, 0, 255))
# orb_kps_img2 = cv.drawKeypoints(
#     img2, keypoints=orb_kps2, outImage=None, color=(0, 0, 255))

# plt.figure(figsize=(16, 8))
# plt.subplot(121)
# plt.imshow(orb_kps_img1[:, :, ::-1])
# plt.axis('off')
# plt.subplot(122)
# plt.imshow(orb_kps_img2[:, :, ::-1])
# plt.axis('off')
# plt.show()


img1 = cv.imread('feature2.jpg')
img2 = cv.imread('feature1.jpg')
orb = cv.ORB_create()
gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
orb_kps1 = orb.detect(gray1, None)
orb_kps2 = orb.detect(gray2, None)

# 묘사자 만들기
orb_kps1, orb_des1 = orb.compute(gray1, orb_kps1)
orb_kps2, orb_des2 = orb.compute(gray2, orb_kps2)

normType = cv.NORM_HAMMING
crossCheck = True

bf = cv.BFMatcher(normType=normType, crossCheck=crossCheck)

matches = bf.match(orb_des1, orb_des2)

matchColor = (0, 0, 255)
singlePointColor = (0, 255, 0)
flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS

match_img = cv.drawMatches(img1, orb_kps1,
                           img2, orb_kps2,
                           matches,
                           outImg=None,
                           matchColor=matchColor, singlePointColor=singlePointColor,
                           flags=flags)

# plt.figure(figsize=(16, 8))
# plt.title('ORB feature matching - Brute-force')
# plt.imshow(match_img[:, :, ::-1])
# plt.axis('off')
# plt.show()


pts1 = []
pts2 = []

for i, m in enumerate(matches):
    pts1.append(orb_kps1[m.queryIdx].pt)
    pts2.append(orb_kps2[m.trainIdx].pt)

pts1 = np.array(pts1, dtype=np.float32)
pts2 = np.array(pts2, dtype=np.float32)

method = cv.FM_LMEDS
ransacReprojThreshold = 3
confidence = 0.99

F, mask = cv.findFundamentalMat(np.int32(pts1), np.int32(pts2),
                                method=method, ransacReprojThreshold=ransacReprojThreshold, confidence=confidence)

pts1_F = pts1[mask.ravel() == 1]
pts2_F = pts2[mask.ravel() == 1]

img = np.concatenate([img1, img2], axis=1)
# plt.figure(figsize=(16, 8))
# plt.title('SIFT feature matching - Fundamental')
# plt.imshow(img[:, :, ::-1])
# plt.plot([pts1_F[:, 0], 726+pts2_F[:, 0]],
#          [pts1_F[:, 1], pts2_F[:, 1]], 'r-', alpha=0.5)
# plt.axis('off')
# plt.show()

focal = 640.0
pp = (320, 240)

method = cv.LMEDS
prob = 0.99
threshold = 1.0

E, mask = cv.findEssentialMat(
    pts2, pts1, focal=focal, pp=pp, method=method, prob=prob, threshold=threshold)

pts1_E = pts1[mask.ravel() == 1]
pts2_E = pts2[mask.ravel() == 1]

img = np.concatenate([img1, img2], axis=1)

# plt.figure(figsize=(16, 8))
# plt.title('ORB feature matching - Essential')
# plt.imshow(img[:, :, ::-1])
# plt.plot([pts1_E[:, 0], 726+pts2_E[:, 0]],
#          [pts1_E[:, 1], pts2_E[:, 1]], 'r-', alpha=0.5)
# plt.axis('off')
# plt.show()

n_inliers, R, t, mask = cv.recoverPose(
    E, pts2, pts1, focal=focal, pp=pp, mask=mask)

print('# of inliers', n_inliers)
print('Rotation matrix', R.shape)
print(R)
print('Translation', t.shape)
print(t)
pts1_P = pts1[mask.ravel() == 1]
pts2_P = pts2[mask.ravel() == 1]
img = np.concatenate([img1, img2], axis=1)

plt.figure(figsize=(16, 8))
plt.title('ORB feature matching - Pose')
plt.imshow(img[:, :, ::-1])
plt.plot([pts1_P[:, 0], 726+pts2_P[:, 0]],
         [pts1_P[:, 1], pts2_P[:, 1]], 'r-', alpha=0.5)
plt.axis('off')
plt.show()


def draw_img_3dax(ax, img, R=None, t=None):
    h, w = img.shape[:2]
    img = img[::-1, :, :]
    xmax = w/np.maximum(h, w)
    ymax = h/np.maximum(h, w)

    xval = np.linespace(-xmax/2.0, xmax/2.0, w)
    yval = np.linespace(-ymax/2.0, ymax/2.0, h)

    xx, yy = np.meshgrid(xval, yval)
    zz = np.zeros(xx.shape)

    if not R is None:
        pts = np.stack([xx, yy, zz], axis=-1)
        pts = np.expand_dims(pts, axis=-1)
        pts = np.matmul(R, pts)
        pts = np.squeeze(pts, axis=-1)
        xx, yy, zz = pts[:, :, 0], pts[:, :, 1], pts[:, :, 2]
    if not t is None:
        xx += t[0]
        yy += t[1]
        zz += t[2]
    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1,
                    facecolors=img, shade=False)


def draw_axis_3dax(ax, R, t, scale=0.1):
    xaxis = t + R[:, 0]*scale
    yaxis = t + R[:, 1]*scale
    zaxis = t + R[:, 2]*scale

    ax.plot([t[0], xaxis[0]], [t[1], xaxis[1]], [t[2], xaxis[2]], 'r-')
    ax.plot([t[0], yaxis[0]], [t[1], yaxis[1]], [t[2], yaxis[2]], 'g-')
    ax.plot([t[0], zaxis[0]], [t[1], zaxis[1]], [t[2], zaxis[2]], 'b-')


h, w = img1.shape[:2]
xmax = w/np.maximum(h, w)
ymax = h/np.maximum(h, w)

img1_n = cv.resize(img1, dsize=(40, 30))/255.0
img2_n = cv.resize(img2, dsize=(40, 30))/255.0

fig = plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.imshow(img1[:, :, ::-1], origin='upper', extent=[-xmax, xmax, -ymax, ymax])
plt.axis('off')
plt.subplot(122)
plt.imshow(img2[:, :, ::-1], origin='upper', extent=[-xmax, xmax, -ymax, ymax])
plt.axis('off')
plt.show()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=-60, azim=-90)
draw_axis_3dax(ax, R=np.eye(3), t=np.array([0, 0, 0]))
draw_axis_3dax(ax, R=R, t=np.squeeze(1.0*t, axis=1))

ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.set_zlim(0.0, 1.0)

plt.show()
