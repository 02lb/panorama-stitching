import numpy as np
import cv2
import harries
import match
import sift
import ransac
import hog

FEATURE_TYPE = 'hog'   # sift or hog

if __name__ == '__main__':
    image1_path = 'images/uttower1.jpg'
    image2_path = 'images/uttower2.jpg'
    save1_path = 'results/uttower1_keypoints.jpg'
    save2_path = 'results/uttower2_keypoints.jpg'
    # 角点检测
    print("------------Harris corner detection------------")
    # 收集返回的角点坐标
    corner_location1 = harries.harris_corner_detection_cv2Method(image1_path, save1_path)
    corner_location2 = harries.harris_corner_detection_cv2Method(image2_path, save2_path)
    print("corner_location1_shape:", corner_location1.shape) # (6398, 2)
    print("corner_location2_shape:", corner_location2.shape) # (8509, 2)

    if FEATURE_TYPE == 'sift':
        # 使用SIFT描述子提取关键点特征
        print("------------SIFT feature extraction------------")
        keypoints1, descriptors1 = sift.sift_feature_extraction(image1_path, corner_location1,)
        keypoints2, descriptors2 = sift.sift_feature_extraction(image2_path, corner_location2)
        print("kp_descriptor1_shape:", descriptors1.shape)  # (6398, 128)
        print("kp_descriptor2_shape:", descriptors2.shape)  # (8509, 128)

        # 使用欧几里得距离作为特征之间相似度的度量，并绘制两幅图像之间的关键点匹配的情况
        print("------------Feature matching------------")
        matchs = match.match_and_draw(image1_path, image2_path, keypoints1, descriptors1, keypoints2, descriptors2, 'results/uttower_match_sift.png')
        print("len_matches: ", len(matchs))

        # 使用RANSAC求解仿射变换矩阵并进行图像拼接
        print("------------Image stitching------------")
        result = ransac.stitch_images(image1_path, image2_path, keypoints1, keypoints2, matchs, FEATURE_TYPE, "results/uttower_stitching_sift.png")
        print("The result of image stitching is saved in 'results/uttower_stitching_sift.png'")

    elif FEATURE_TYPE == 'hog':
        # 使用HOG描述子提取关键点特征
        print("------------HOG feature extraction------------")
        keypoints1_hog, descriptors1_hog = hog.hog_feature_extraction(image1_path, corner_location1)
        keypoints2_hog, descriptors2_hog = hog.hog_feature_extraction(image2_path, corner_location2)
        print("kp_descriptor1_hog_shape:", descriptors1_hog.shape)
        print("kp_descriptor2_hog_shape:", descriptors2_hog.shape)

        # 使用欧几里得距离作为特征之间相似度的度量，并绘制两幅图像之间的关键点匹配的情况
        print("------------Feature matching------------")
        matches_hog = hog.hog_match_and_draw(image1_path, image2_path, keypoints1_hog, descriptors1_hog, keypoints2_hog,
                                           descriptors2_hog)

        # 使用RANSAC求解仿射变换矩阵并进行图像拼接
        print("------------Image stitching------------")
        result = ransac.stitch_images(image1_path, image2_path, keypoints1_hog, keypoints2_hog, matches_hog, FEATURE_TYPE, "results/uttower_stitching_hog.png")

    else:
        print("Please choose the correct FEATURE_TYPE: 'sift' or 'hog'")
















