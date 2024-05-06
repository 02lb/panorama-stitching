import cv2
import numpy as np

def hog_feature_extraction(image_path, corner_location):
    gray = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
    print("corner_location_shape:", corner_location.shape)
    # 转换角点坐标为元组
    corner_location_tuple = [(int(x[1]), int(x[0])) for x in corner_location]
    hog = cv2.HOGDescriptor()
    kp_hog = [cv2.KeyPoint(float(y), float(x), 1) for x, y in corner_location]

    des_hog = hog.compute(gray, locations=corner_location_tuple).reshape(corner_location.shape[0], -1)
    return kp_hog, des_hog


def hog_match_and_draw(image1_path, image2_path, keypoints1, descriptors1, keypoints2, descriptors2):
    # 使用欧氏距离计算特征之间的相似度
    bf = cv2.BFMatcher(cv2.NORM_L2)
    # matches = bf.match(descriptors1, descriptors2)
    # 使用knnmatch
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    print("hog_match_out_len", len(matches))

    cnt1, cnt2 = 0, 0
    for d1 in descriptors1:
        if d1.sum() == 0:
            cnt1 += 1
    for d2 in descriptors2:
        if d2.sum() == 0:
            cnt2 += 1
    print("descriptors1 all zero:", cnt1)
    print("descriptors2 all zero:", cnt2)

    # 选择最佳匹配
    good = []
    for m, n in matches:
        if m.distance < 0.95 * n.distance:
            good.append(m)
    matches = good
    print("hog_match_out_len", len(matches))

    # # 按照距离进行排序
    # matches = sorted(matches, key=lambda x: x.distance)[:100]
    # print(matches[0].trainIdx, matches[0].queryIdx, matches[0].distance) # 0 185 0.0
    # print(len(matches), matches[0].distance, matches[-1].distance)

    # 绘制关键点匹配结果
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    match_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 保存关键点匹配结果
    cv2.imwrite('results/uttower_match_hog.png', match_img)

    return matches