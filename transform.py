import numpy as np
import cv2

def construct_homogeneous_matrix(R, t):
    """
    构造齐次变换矩阵
    :param R: 旋转矩阵 (3x3)
    :param t: 平移向量 (3x1)
    :return: 齐次变换矩阵 (4x4)
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()  # 平移向量
    return T

def pose_to_homogeneous_matrix(xyz, rxyz):
    """
    将末端执行器的 (x, y, z, rx, ry, rz) 转换为齐次变换矩阵
    :param xyz: 位置 (x, y, z)
    :param rxyz: 旋转向量 (rx, ry, rz)
    :return: 齐次变换矩阵 (4x4)
    """
    x, y, z = xyz
    
    rvec = np.array(rxyz, dtype=np.float32)  # 旋转向量
    R, _ = cv2.Rodrigues(rvec)  # 旋转矩阵
    
    T = np.eye(4)
    T[:3, :3] = R  # 旋转矩阵
    T[:3, 3] = [x, y, z]  # 平移向量
    
    return T

# 1. 相机内参矩阵 (3x3)
matrixCamera2Pixel = np.array([[607.376220703125, 0, 326.4593200683594],
                          [0, 606.376953125, 241.86891174316406],
                          [0, 0, 1]], dtype=np.float32)

# 2. 畸变系数
dist_coeffs = np.array([0,0,0,0,0], dtype=np.float32)

# 3. 2D图像坐标和深度 
u = 320
v = 240 
depth = 0.271

# 4. 手眼标定结果: 旋转矩阵和平移向量
R_cam_to_gripper = np.array([[-0.7869618, -0.9967753, 0.01568168], [0.9967924, -0.07890772, -0.0133601], [0.01455442, 0.01458, 0.99978777]], dtype=np.float32) 
t_cam_to_gripper = np.array([[0.09623856], [-0.02799924], [0.0338227]], dtype=np.float32)

# 5. 末端执行器的 (xyz) 和 (rx, ry, rz) 
gripper_xyz = [-0.365464, -0.092766, 0.374136]  # 末端执行器的位置
gripper_rxyz = [-0.238, -0.337, 3.147]   # 末端执行器的旋转向量

# 构造手眼标定矩阵 (4x4)
matrixHand2Camera = construct_homogeneous_matrix(R_cam_to_gripper, t_cam_to_gripper)

# 构造末端执行器的当前位姿 (4x4)
matrixBase2Hand = pose_to_homogeneous_matrix(gripper_xyz, gripper_rxyz)

# 计算 Base 到 Camera 的变换矩阵
matrixBase2Camera = np.dot(matrixBase2Hand, matrixHand2Camera)

# 计算 Camera 到 Base 的变换矩阵
matrixCamera2Base = np.linalg.inv(matrixBase2Camera)

# 畸变校正，转换像素坐标到归一化图像坐标系
undistorted_coords = cv2.undistortPoints(np.array([[u, v]], dtype=np.float32), matrixCamera2Pixel, dist_coeffs)

# 归一化图像坐标 (xn, yn)
xn, yn = undistorted_coords[0, 0]

# 根据深度信息，计算相机坐标系下的3D点
camera_coords = np.array([xn * depth, yn * depth, depth])

# 计算3D点在 Base 坐标系下的位置
outputBase2 = np.dot(np.linalg.inv(matrixCamera2Base[0:3, 0:3]), camera_coords.reshape(3, 1) - matrixCamera2Base[:3, 3].reshape(3, 1))

print(f"目标点坐标: ", outputBase2.flatten())

