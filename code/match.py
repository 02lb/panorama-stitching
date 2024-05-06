import cv2
import numpy as np


def match_and_draw(image1_path, image2_path, keypoints1, descriptors1, keypoints2, descriptors2, save_path=None):
    # 使用Brute-Force匹配器，对特征点进行匹配
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    print("bf_matcher_out_len", len(matches))

    # 根据距离排序，只保留前n个匹配
    n = 100 # 拼接需要的匹配点数该如何选取：按照距离阈值？
    matches = sorted(matches, key = lambda x:x.distance)[:n]
    print("out_max_distance", matches[n-1].distance)

    # 绘制匹配结果
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 保存匹配结果
    match_save_path = save_path
    if match_save_path:
        cv2.imwrite(match_save_path, img_matches)


    print("match_save_path:", match_save_path)

    # 返回匹配点
    return matches
