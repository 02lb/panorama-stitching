import cv2
import numpy as np

def stitch_images(image1_path, image2_path, keypoints1, keypoints2, matches, feature_type, save_path=None):
    # 读取图像
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # 获取匹配的关键点
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 计算透视变换矩阵
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # 对图像进行透视变换
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    corners1 = np.float32([[0, 0], [0, h1 - 1], [w1 - 1, h1 - 1], [w1 - 1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2 - 1], [w2 - 1, h2 - 1], [w2 - 1, 0]]).reshape(-1, 1, 2)
    corners2_transformed = cv2.perspectiveTransform(corners2, M)
    corners = np.concatenate((corners1, corners2_transformed), axis=0)
    [x_min, y_min] = np.int32(corners.min(axis=0).ravel() - 1)
    [x_max, y_max] = np.int32(corners.max(axis=0).ravel() + 1)

    # 计算平移矩阵，确保第二幅图像位置合适
    translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    M = np.dot(translation_matrix, M)


    # 进行透视变换
    result = cv2.warpPerspective(image2, M, (x_max - x_min, y_max - y_min))
    print("result_shape:", result.shape)
    print("image1_shape:", image1.shape)
    print(-y_min, h1 - y_min, -x_min, w1 - x_min)
    print(result[-y_min:h1 - y_min, -x_min:w1 - x_min].shape)
    result[-y_min:h1 - y_min, -x_min:w1 - x_min] = image1


    print(result.shape)

    # 保存拼接后的全景图
    cv2.imwrite(save_path, result)

    return result

