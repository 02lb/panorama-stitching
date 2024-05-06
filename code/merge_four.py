import cv2
import numpy as np
import harries
import sift
import ransac
import match

def merge_two(img_path1, img_path2):
    # 使用Harris角点检测算法提取关键点
    keypoints1 = harries.harris_corner_detection_cv2Method(img_path1, None)
    keypoints2 = harries.harris_corner_detection_cv2Method(img_path2, None)
    print("keypoints1_shape:", keypoints1.shape)
    print("keypoints2_shape:", keypoints2.shape)

    # 使用SIFT算法计算特征描述子
    keypoints1, descriptors1 = sift.sift_feature_extraction(img_path1, keypoints1)
    keypoints2, descriptors2 = sift.sift_feature_extraction(img_path2, keypoints2)
    print("descriptors1_shape:", descriptors1.shape)
    print("descriptors2_shape:", descriptors2.shape)

    # 使用欧氏距离计算特征之间的相似度
    matches = match.match_and_draw(img_path1, img_path2, keypoints1, descriptors1, keypoints2, descriptors2, None)
    print("len_matches: ", len(matches))

    # 使用RANSAC求解仿射变换矩阵并进行图像拼接
    result = ransac.stitch_images(img_path1, img_path2, keypoints1, keypoints2, matches, 'sift', "results/yosemite_stitching.png")
    print("The result of image stitching is saved in 'results/yosemite_stitching.png'")
    result_path = "results/yosemite_stitching.png"

    return result_path

if __name__ == '__main__':
    img_path1 = 'images/yosemite1.jpg'
    img_path2 = 'images/yosemite2.jpg'
    img_path3 = 'images/yosemite3.jpg'
    img_path4 = 'images/yosemite4.jpg'


    result_path = merge_two(img_path1, img_path2)
    result_path = merge_two(img_path3, result_path)
    result_path = merge_two(img_path4, result_path)