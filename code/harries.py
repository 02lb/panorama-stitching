import cv2
import numpy as np


def harris_corner_detection(image_path, save_path):
    """
    手动实现Harris角点检测算法
    """
    # 导入图像
    img = cv2.imread(image_path)
    # print(img.shape) # (563, 558, 3)三通道
    # 由于Harris角点检测算法只能处理灰度图像，因此需要将图像转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray.shape) # (563, 558)

    # 计算梯度
    # 使用Sobel算子计算图像的梯度，得到图像在x和y方向上的梯度
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # print(dx.shape) # (563, 558)

    # 计算自相关矩阵 M
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    k = 0.04
    # 使用高斯滤波器对自相关矩阵进行平滑处理
    M = cv2.GaussianBlur(Ixx, (5, 5), 0) * cv2.GaussianBlur(Iyy, (5, 5), 0) - cv2.GaussianBlur(Ixy, (5, 5), 0)**2

    # 计算角点响应函数 R
    R = M - k * (Ixx + Iyy)**2

    # 非极大值抑制
    R_max = np.max(R)
    threshold = 0.05 * R_max # 设定阈值为最大响应值的2%
    corner_img = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if R[i, j] > threshold:
                # 画出角点，参数分别为图像，圆心坐标，半径，颜色，线宽
                cv2.circle(corner_img, (j, i), 1, (255,0,255), -1)

    # 保存结果
    if save_path:
        cv2.imwrite(save_path, corner_img)

    # 返回角点坐标 N*2
    return np.argwhere(R > threshold)




def harris_corner_detection_cv2Method(image_path, save_path):
    """
    使用cv2.cornerHarris()方法计算角点响应,对比自己实现的Harris角点检测算法效果
    """
    # 导入图像
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用cv2.cornerHarris()方法计算角点响应
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # 膨胀角点响应结果以便更好的显示
    #dst = cv2.dilate(dst, None)

    # 设定阈值，筛选出角点
    img[dst > 0.05 * dst.max()] = [255, 0, 255]

    # 保存结果
    if save_path:
        cv2.imwrite(save_path, img)

    # 返回角点坐标 N*2
    return np.argwhere(dst > 0.01 * dst.max())


if __name__ == '__main__':
    # 运行角点检测
    harris_corner_detection('images/sudoku.png', 'results/sudoku_keypoints.png')
    harris_corner_detection_cv2Method('images/sudoku.png', 'results/sudoku_keypoints_cv2Method.png')

    # # Mission 2:
    # harris_corner_detection('images/uttower1.jpg', 'results/uttower1_keypoints.jpg')
    # harris_corner_detection_cv2Method('images/uttower1.jpg', 'results/uttower1_keypoints_cv2Method.jpg')
    # harris_corner_detection('images/uttower2.jpg', 'results/uttower2_keypoints.jpg')
    # harris_corner_detection_cv2Method('images/uttower2.jpg', 'results/uttower2_keypoints_cv2Method.jpg')