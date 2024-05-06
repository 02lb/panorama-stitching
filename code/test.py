import cv2
import numpy as np


# 读取两幅图像
img1 = cv2.imread('./images/uttower1.jpg')
img2 = cv2.imread('./images/uttower2.jpg')

# 转换为灰度图像
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 使用 Harris 算法检测关键点
harris_corners1 = cv2.cornerHarris(gray1, 2, 3, 0.04)
harris_corners2 = cv2.cornerHarris(gray2, 2, 3, 0.04)

y, x = np.where(harris_corners1.astype(float) > 0.01 * harris_corners1.max())
kp1 = [(x[i], y[i]) for i in range(len(x))]

y, x = np.where(harris_corners2.astype(float) > 0.01 * harris_corners2.max())
kp2 = [(x[i], y[i]) for i in range(len(x))]

y, x = np.where(harris_corners1.astype(float) > 0.01 * harris_corners1.max())
kkp1 = [cv2.KeyPoint(x[i].astype(float), y[i].astype(float), 1) for i in range(len(x))]

y, x = np.where(harris_corners2.astype(float) > 0.01 * harris_corners2.max())
kkp2 = [cv2.KeyPoint(x[i].astype(float), y[i].astype(float), 1) for i in range(len(x))]

# 计算 HOG 特征
hog = cv2.HOGDescriptor()
desc1 = hog.compute(gray1, locations=kp1).reshape(len(kp1), -1)
desc2 = hog.compute(gray2, locations=kp2).reshape(len(kp2), -1)

# 使用 BFMatcher 和欧式距离进行特征匹配
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(desc1, desc2, k=2)

# 应用 Lowe's ratio test 来过滤匹配结果
good_matches = []
for m, n in matches:
    if m.distance < 0.9 * n.distance:
        good_matches.append(m)

# print(good_matches)
# 绘制匹配结果
img3 = cv2.drawMatches(img1, kkp1, img2, kkp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('./results/uttower_match_hog.jpg', img3)
cv2.imshow('Matches', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()