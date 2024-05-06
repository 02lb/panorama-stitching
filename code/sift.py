import cv2
import numpy as np
from matplotlib import pyplot as plt


# 根据已经有的角点坐标进行SIFT特征提取
def sift_feature_extraction(image_path, corner_location):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用SIFT算法提取特征 corner_location:(N, 2)
    sift = cv2.SIFT_create()
    # cv2.KeyPoint： 它使用角点的坐标(x[0], x[1])创建一个cv2.KeyPoint对象，第三个参数5表示关键点的大小（即角点的邻域大小）。
    keypoints = [cv2.KeyPoint(float(x[1]), float(x[0]), 5) for x in corner_location]
    keypoints, descriptors = sift.compute(gray_img, keypoints)
    # print("sift keypoints shape:", keypoints.shape)
    # print("sift descriptors shape:", descriptors.shape)
    return keypoints, descriptors